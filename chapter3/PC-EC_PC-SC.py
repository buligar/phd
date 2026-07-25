"""
PC-SC vs PC-EC comparison for two reference dynamics:

1) Lorenz attractor, D=3.
2) Ordinary 2D oscillator, D=2:
       theta = omega * dt
       M = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
       u[n] = M @ u[n-1]

Architectures:
- PC-EC: latent state z is driven directly by continuous error e = o1 - g.
- PC-SC: error e is encoded by a separate spiking error population, decoded as e_dec,
       and e_dec drives latent state z.

Outputs:
- sweep over N_sens = 100..1000 with step 100;
- synchronization quality between autonomous o1 and top-down prediction g;
- runtime and memory metrics;
- overview figures:
    row 1: reference, autonomous o1, g, latent state z
    row 2: error, spikes o1, spikes error population only for PC-SC, spikes latent z/o2
- component-signal figures: reference vs autonomous o1 vs g;
- summary CSV and comparison figures.

Important:
The error spiking raster is shown only for PC-SC, because only PC-SC has a real
spiking error population. PC-EC uses a continuous error signal, so no
error-spike monitor or fake error raster is created.
"""

import os
import gc
import time
import math
import threading
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MaxNLocator, ScalarFormatter
try:
    import psutil
except ImportError:
    psutil = None


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # Which examples to run. Use ("oscillator",) for oscillator only.
    # signal_names: tuple = ("lorenz", "oscillator")
    # signal_names: tuple = ("oscillator",)
    signal_names: tuple = ("lorenz",)

    # Main sweep
    n_sens_values: tuple = tuple(range(540, 541, 10))

    # Common simulation
    T: float = 30.0
    dt: float = 0.001

    # Lorenz parameters
    rho: float = 28.0
    sigma: float = 10.0
    beta: float = 2.667
    x0_lorenz: tuple = (1.0, 1.0, 1.0)

    # Oscillator parameters
    oscillator_omega: float = 2.0 * math.pi  # rad/s, 1 Hz
    x0_oscillator: tuple = (1.0, 0.0)

    # NEF/LIF
    tau_syn: float = 0.05
    tau_rc: float = 0.02
    tau_ref: float = 0.002
    tau_o1: float = 0.05
    tau_z: float = 0.05
    rate_low: float = 100.0
    rate_high: float = 200.0

    # Training / learning
    cue_end: float = 5
    ridge_lambda_sens: float = 1e-2
    ridge_lambda_dec: float = 1e-2
    eta: float = 1e-3

    # In the old code z_dot = -z/tau_z + e.
    # This makes steady z about tau_z * e, often too small.
    # k_error_to_z = 1/tau_z makes z track the error scale with time constant tau_z.
    # To reproduce the old behavior, set k_error_to_z = 1.0.
    k_error_to_z: float = 1.0

    # Fixed hidden populations
    n_lat: int = 50
    n_err: int = 50

    # Offline recurrent decoder for autonomous o1.
    # Larger values can improve quality but increase RAM/time.
    max_train_samples: int = 3000

    # Static decoders for PC-SC error population
    decoder_train_samples: int = 6000

    # Metrics
    sync_start_offset_after_cue: float = 0.5
    smooth_win: int = 101
    amp_thresh: float = 1e-3

    # Plotting
    out_dir: str = "results_PC-SC_nePC-SC_lorenz_oscillator"
    dpi: int = 220
    save_each_n_figures: bool = True
    save_comparison_figures: bool = True
    show_plots: bool = False
    plot_from_sec: float = 0
    spike_plot_max_neurons: int = 200

    # Randomness
    base_seed: int = 1



CFG = Config()

mpl.rcParams.update({
    # Общий стиль под журнальную фигуру
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})


# =============================================================================
# Timing and memory
# =============================================================================

class PeakRSSMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak = np.nan
        self.start_rss = np.nan
        self.end_rss = np.nan
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def start(self):
        if self._process is None:
            return
        self.start_rss = self._process.memory_info().rss
        self.peak = self.start_rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self.peak:
                    self.peak = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        if self._process is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_rss = self._process.memory_info().rss

    @property
    def peak_delta_mb(self):
        if not np.isfinite(self.peak) or not np.isfinite(self.start_rss):
            return np.nan
        return (self.peak - self.start_rss) / (1024 ** 2)

    @property
    def rss_delta_mb(self):
        if not np.isfinite(self.end_rss) or not np.isfinite(self.start_rss):
            return np.nan
        return (self.end_rss - self.start_rss) / (1024 ** 2)


def measure_call(func, *args, **kwargs):
    gc.collect()
    monitor = PeakRSSMonitor(interval=0.005)
    tracemalloc.start()
    monitor.start()
    t0 = time.perf_counter()
    result = None
    try:
        result = func(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - t0
        monitor.stop()
        _, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    stats = {
        "time_s": elapsed,
        "python_peak_mem_mb": py_peak / (1024 ** 2),
        "rss_peak_delta_mb": monitor.peak_delta_mb,
        "rss_delta_mb": monitor.rss_delta_mb,
    }
    return result, stats


# =============================================================================
# LIF / NEF helpers
# =============================================================================

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def current_from_rate(r, tau_rc=0.02, tau_ref=0.002):
    r = np.asarray(r, dtype=float)
    k = (1.0 / r - tau_ref) / tau_rc
    return 1.0 / (1.0 - np.exp(-k))


def lif_rate_from_current(J, tau_rc=0.02, tau_ref=0.002):
    J = np.asarray(J, dtype=float)
    out = np.zeros_like(J, dtype=float)
    mask = J > 1.0
    out[mask] = 1.0 / (tau_ref - tau_rc * np.log1p(-1.0 / J[mask]))
    return out


def init_population(N, D, rng, tau_rc, tau_ref, rate_low=100.0, rate_high=200.0):
    enc = rng.normal(size=(N, D))
    enc /= np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12

    x_intercepts = rng.uniform(-1.0, 1.0, size=N)
    max_rates = rng.uniform(rate_low, rate_high, size=N)

    Jmax = current_from_rate(max_rates, tau_rc, tau_ref)
    gain = (Jmax - 1.0) / (1.0 - x_intercepts)
    bias = 1.0 - gain * x_intercepts
    return enc.astype(float), gain.astype(float), bias.astype(float)


def lif_population_step(V, ref_count, inp, enc, gain, bias, dt, tau_rc, tau_ref):
    J = gain * (enc @ inp) + bias
    active = ref_count <= 0

    V[active] += (dt / tau_rc) * (-V[active] + J[active])
    V[~active] = 0.0

    spikes = V >= 1.0
    V[spikes] = 0.0

    ref_count[ref_count > 0] -= 1
    ref_steps = max(1, int(round(tau_ref / dt)))
    ref_count[spikes] = ref_steps

    return V, ref_count, spikes


def update_filtered_activity(a, spikes, dt, tau_syn):
    return a + dt * (-a / tau_syn) + spikes.astype(float) / tau_syn


def population_rates(X, enc, gain, bias, tau_rc, tau_ref):
    J = X @ enc.T
    J = J * gain[None, :] + bias[None, :]
    return lif_rate_from_current(J, tau_rc=tau_rc, tau_ref=tau_ref)

    
def solve_ridge(A, Y, lam=1e-2):
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    G = A.T @ A
    B = A.T @ Y
    # Регуляризация относительно масштаба активности, а не абсолютная.
    # Иначе фиксированная lam становится пренебрежимой при большом N ->
    # переобучение рекуррентного декодера -> срыв предельного цикла o1.
    reg = lam * np.mean(np.diag(G))
    return np.linalg.solve(G + reg * np.eye(G.shape[0]), B)


def solve_static_decoder(enc, gain, bias, D, rng, cfg: Config, target="identity"):
    X = rng.uniform(-1.0, 1.0, size=(cfg.decoder_train_samples, D))
    n_zero = min(cfg.decoder_train_samples, max(100, cfg.decoder_train_samples // 10))
    X[:n_zero] = 0.25 * rng.normal(size=(n_zero, D))
    X = np.clip(X, -1.0, 1.0)
    A = population_rates(X, enc, gain, bias, cfg.tau_rc, cfg.tau_ref)
    if target == "identity":
        Y = X
    else:
        raise ValueError(f"Unknown decoder target: {target}")
    return solve_ridge(A, Y, cfg.ridge_lambda_dec)


# =============================================================================
# Reference dynamics
# =============================================================================

def lorenz_rk4_trajectory(T, dt, x0, sigma=10.0, rho=28.0, beta=2.667):
    t = np.arange(0.0, T + dt, dt)
    N_t = t.size
    raw = np.zeros((N_t, 3), dtype=float)
    raw[0] = np.asarray(x0, dtype=float)

    def f(s):
        x, y, z = s
        return np.array([
            sigma * (y - x),
            x * rho - x * z - y,
            x * y - beta * z,
        ], dtype=float)

    s = raw[0].copy()
    for n in range(1, N_t):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        raw[n] = s

    scale = np.max(np.abs(raw), axis=0)
    ref = raw / (scale + 1e-12)
    return t, raw, ref


def oscillator_trajectory(T, dt, omega=2.0 * math.pi, x0=(1.0, 0.0)):
    """Ordinary 2D oscillator exactly as a rotation map."""
    t = np.arange(0.0, T + dt, dt)
    N_t = t.size
    theta = omega * dt
    M = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=float)

    u = np.zeros((N_t, 2), dtype=float)
    u[0] = np.asarray(x0, dtype=float)
    u[0] /= np.linalg.norm(u[0]) + 1e-12

    for n in range(1, N_t):
        u[n] = M @ u[n - 1]

    raw = u.copy()
    ref = u.copy()
    return t, raw, ref


def make_reference_signal(signal_name, cfg: Config):
    name = signal_name.lower().strip()
    if name == "lorenz":
        t, raw, ref = lorenz_rk4_trajectory(
            cfg.T, cfg.dt, cfg.x0_lorenz,
            sigma=cfg.sigma, rho=cfg.rho, beta=cfg.beta,
        )
        return {
            "signal": "lorenz",
            "title": "Lorenz attractor",
            "t": t,
            "raw": raw,
            "ref": ref,
            "D": 3,
            "labels": ["x", "y", "z"],
        }
    if name == "oscillator":
        t, raw, ref = oscillator_trajectory(
            cfg.T, cfg.dt,
            omega=cfg.oscillator_omega,
            x0=cfg.x0_oscillator,
        )
        return {
            "signal": "oscillator",
            "title": "Ordinary oscillator",
            "t": t,
            "raw": raw,
            "ref": ref,
            "D": 2,
            "labels": ["x", "y"],
        }
    raise ValueError(f"Unknown signal_name={signal_name!r}. Use 'lorenz' or 'oscillator'.")


# =============================================================================
# Synchronization metrics
# =============================================================================

def analytic_signal(x):
    x = np.asarray(x, dtype=float)
    N = x.size
    Xf = np.fft.fft(x)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(Xf * H)


def moving_average(x, win):
    x = np.asarray(x, dtype=float)
    k = int(win)
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def sync_metrics_1d(ref, rec, dt, smooth_win=1001, amp_thresh=1e-3):
    ref = np.asarray(ref, dtype=float)
    rec = np.asarray(rec, dtype=float)

    ref_c = ref - np.mean(ref)
    rec_c = rec - np.mean(rec)

    z_ref = analytic_signal(ref_c)
    z_rec = analytic_signal(rec_c)

    A_ref = np.abs(z_ref)
    A_rec = np.abs(z_rec)
    amp_corr = safe_corr(A_ref, A_rec)

    mask = (A_ref > amp_thresh) & (A_rec > amp_thresh)

    phi_ref = np.unwrap(np.angle(z_ref))
    phi_rec = np.unwrap(np.angle(z_rec))

    f_ref = np.diff(phi_ref) / (2.0 * np.pi * dt)
    f_rec = np.diff(phi_rec) / (2.0 * np.pi * dt)

    if smooth_win is not None and smooth_win > 1:
        f_ref = moving_average(f_ref, smooth_win)
        f_rec = moving_average(f_rec, smooth_win)

    mask_f = mask[1:]
    if np.sum(mask_f) > 2:
        freq_corr = safe_corr(f_ref[mask_f], f_rec[mask_f])
    else:
        freq_corr = np.nan

    dphi = (phi_ref - phi_rec)[mask]
    if dphi.size:
        c = np.mean(np.exp(1j * dphi))
        plv = float(np.abs(c))
        mean_phase_diff = float(np.angle(c))
    else:
        plv = np.nan
        mean_phase_diff = np.nan

    return amp_corr, freq_corr, plv, mean_phase_diff


def summarize_sync(u, g, ref, t, cfg: Config, labels):
    start = cfg.cue_end + cfg.sync_start_offset_after_cue
    idx = int(np.searchsorted(t, start))
    if idx >= len(t) - 4:
        idx = max(0, len(t) // 2)

    D = ref.shape[1]
    out = {}
    amp, freq, plv = [], [], []
    for d in range(D):
        lab = labels[d]
        a, f, p, ph = sync_metrics_1d(
            u[idx:, d],
            g[idx:, d],
            cfg.dt,
            smooth_win=cfg.smooth_win,
            amp_thresh=cfg.amp_thresh,
        )
        out[f"amp_corr_{lab}"] = a
        out[f"freq_corr_{lab}"] = f
        out[f"plv_{lab}"] = p
        out[f"mean_phase_deg_{lab}"] = np.degrees(ph) if np.isfinite(ph) else np.nan
        amp.append(a)
        freq.append(f)
        plv.append(p)

    out["mean_amp_corr"] = float(np.nanmean(amp))
    out["mean_freq_corr"] = float(np.nanmean(freq))
    out["mean_plv"] = float(np.nanmean(plv))
    out["rmse_o1_g"] = float(np.sqrt(np.mean((u[idx:] - g[idx:]) ** 2)))
    out["rmse_ref_o1"] = float(np.sqrt(np.mean((ref[idx:] - u[idx:]) ** 2)))
    out["rmse_ref_g"] = float(np.sqrt(np.mean((ref[idx:] - g[idx:]) ** 2)))
    return out


# =============================================================================
# Spike storage and plotting
# =============================================================================

def append_spikes(times_list, ids_list, t_now, spikes, max_neurons=None):
    if not np.any(spikes):
        return
    idx = np.flatnonzero(spikes)
    if max_neurons is not None:
        idx = idx[idx < max_neurons]
    if idx.size == 0:
        return
    times_list.append(np.full(idx.size, t_now, dtype=float))
    ids_list.append(idx.astype(int))


def concat_spikes(times_list, ids_list):
    if len(times_list) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(times_list), np.concatenate(ids_list)


def set_panel_title(ax, panel_label, title, pad=14, label_size=18, title_size=17):
    """
    Ставит букву панели a), b), ... на одну линию с заголовком графика.

    Важно: используется механизм двух title в Matplotlib:
    - loc="left" для буквы панели;
    - loc="center" для основного заголовка.
    Поэтому буква не находится внутри области графика, а стоит на уровне title.
    """
    ax.set_title(title, loc="center", pad=pad, fontsize=title_size)
    ax.set_title(panel_label, loc="left", pad=pad, fontsize=label_size)


def _auto_limits_2d(ax, data):
    data = np.asarray(data)
    x_min, y_min = np.nanmin(data[:, 0]), np.nanmin(data[:, 1])
    x_max, y_max = np.nanmax(data[:, 0]), np.nanmax(data[:, 1])
    dx = max(x_max - x_min, 1e-3)
    dy = max(y_max - y_min, 1e-3)
    ax.set_xlim(x_min - 0.08 * dx, x_max + 0.08 * dx)
    ax.set_ylim(y_min - 0.08 * dy, y_max + 0.08 * dy)

def apply_tick_format_3d(ax, fixed_limits=True):
    """
    Для fixed_limits=True:
        x,y: -1, 0, 1
        z:   0, 0.5, 1

    Для fixed_limits=False:
        автоматические 3 тика, но без отображения масштаба типа 1e3.
    """

    if fixed_limits:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([0, 0.5, 1])
    else:
        ax.set_xlabel("x", labelpad=16)
        ax.set_ylabel("y", labelpad=16)
        ax.set_zlabel("z", labelpad=16)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        axis.set_major_formatter(formatter)

    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.zaxis.get_offset_text().set_visible(False)

def style_phase(ax, data, title, fixed_limits=True):
    """Phase portrait style for both D=2 and D=3."""
    data = np.asarray(data, dtype=float)
    D = data.shape[1]

    if D == 3:
        ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=0.85)

        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("y", labelpad=10)
        ax.set_zlabel("z", labelpad=10)

        ax.tick_params(axis="x", labelsize=16, pad=4)
        ax.tick_params(axis="y", labelsize=16, pad=4)
        ax.tick_params(axis="z", labelsize=16, pad=4)

        ax.view_init(elev=32, azim=-62)
        ax.set_box_aspect((1.0, 1.0, 0.80))

        if fixed_limits:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, 1)

        apply_tick_format_3d(ax, fixed_limits=fixed_limits)

        ax.grid(True, alpha=0.4)
        return

    if D == 2:
        ax.plot(data[:, 0], data[:, 1], linewidth=0.95)
        # Заголовок задаётся отдельно через set_panel_title().
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.tick_params(axis="both", labelsize=16)
        # ax.set_aspect("equal", adjustable="box")
        # if fixed_limits:
        #     ax.set_xlim(-1, 1)
        #     ax.set_ylim(-1, 1)
        # else:
        #     _auto_limits_2d(ax, data)
        # ax.grid(True, linestyle="--", alpha=0.35)
        return

    raise ValueError("Only D=2 or D=3 is supported for phase plotting.")


def add_phase_subplot(fig, gs_cell, data, title, fixed_limits=True, panel_label=None):
    D = np.asarray(data).shape[1]
    if D == 3:
        ax = fig.add_subplot(gs_cell, projection="3d")
    else:
        ax = fig.add_subplot(gs_cell)

    style_phase(ax, data, title, fixed_limits=fixed_limits)

    if panel_label is not None:
        set_panel_title(ax, panel_label, title, pad=14)
    else:
        ax.set_title(title, loc="center", pad=14, fontsize=17)

    return ax


def plot_overview(result, cfg: Config, out_path):
    t = result["t"]
    ref = result["ref"]
    u = result["u"]
    g = result["g"]
    z = result["z"]
    e_true = result["e_true"]
    N_sens = result["N_sens"]
    arch = result["arch"]
    signal = result["signal"]
    signal_title = result["signal_title"]
    labels = result["labels"]

    i0 = int(np.searchsorted(t, cfg.plot_from_sec))

    spk_o1_t, spk_o1_i = result["spikes_o1"]
    spk_err_t, spk_err_i = result["spikes_error"]
    spk_z_t, spk_z_i = result["spikes_z"]

    # Более компактная компоновка: меньше расстояния между панелями,
    # но внутри 3D-графиков сохранён увеличенный labelpad для осей x,y,z.
    fig = plt.figure(figsize=(20, 10.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.3)
    fig.suptitle(
        rf"{signal_title} | {arch}, $N_{{\mathrm{{sens}}}} = {N_sens}$"
    )

    ax1 = add_phase_subplot(
        fig, gs[0, 0], ref[i0:],
        "Reference signal input",
        fixed_limits=True,
        panel_label="a)",
    )

    ax2 = add_phase_subplot(
        fig, gs[0, 1], u[i0:],
        r"Autonomous sensory" + "\n" + r"dynamics $o_1(t)$",
        fixed_limits=True,
        panel_label="b)",
    )

    ax3 = add_phase_subplot(
        fig, gs[0, 2], g[i0:],
        r"Top-down" + "\n" + r" prediction $g(t)$",
        fixed_limits=True,
        panel_label="c)",
    )

    # Do not fix z limits. This shows the real scale of latent z, as in your example.
    ax4 = add_phase_subplot(
        fig, gs[0, 3], z[i0:],
        r"Latent state $z(t)$",
        fixed_limits=False,
        panel_label="d)",
    )

    ax5 = fig.add_subplot(gs[1, 0])
    for d, lab in enumerate(labels):
        ax5.plot(t[i0:], e_true[i0:, d], linewidth=0.9, label=rf"$e_{lab}$")
    ax5.axvline(cfg.cue_end, linestyle="--", linewidth=0.9, color="k", alpha=0.8)
    set_panel_title(ax5, "e)", r"Prediction error", pad=12)
    ax5.set_xlabel("Time, s")
    ax5.tick_params(axis="both", labelsize=16)
    ax5.grid(True, linestyle="--", alpha=0.35)
    ax5.legend(frameon=False, loc="upper right")

    ax6 = fig.add_subplot(gs[1, 1])
    if spk_o1_t.size > 0:
        ax6.plot(spk_o1_t, spk_o1_i, "|", markersize=2.0)
    ax6.axvline(cfg.cue_end, linestyle="--", linewidth=0.9, color="k", alpha=0.8)
    set_panel_title(ax6, "f)", r"Spiking activity $o_1$", pad=12)
    ax6.set_xlabel("Time, s")
    ax6.tick_params(axis="both", labelsize=16)
    ax6.set_ylabel(r"Neurons $o_1$")
    ax6.set_ylim(0, min(N_sens, cfg.spike_plot_max_neurons))
    ax6.grid(True, linestyle="--", alpha=0.35)

    if arch == "PC-SC":
        ax7 = fig.add_subplot(gs[1, 2])
        if spk_err_t.size > 0:
            ax7.plot(spk_err_t, spk_err_i, "|", markersize=2.0)
        ax7.axvline(cfg.cue_end, linestyle="--", linewidth=0.9, color="k", alpha=0.8)
        set_panel_title(ax7, "g)", r"Spiking activity error", pad=12)
        ax7.set_xlabel("Time, s")
        ax7.tick_params(axis="both", labelsize=16)
        ax7.set_ylabel(r"Neurons error")
        ax7.set_ylim(0, cfg.n_err)
        ax7.grid(True, linestyle="--", alpha=0.35)

        ax8 = fig.add_subplot(gs[1, 3])
        panel_label_o2 = "h)"
    else:
        # PC-EC has no spiking error population, so this figure must not
        # show a fake error-spike raster. We place o2 directly after o1.
        ax8 = fig.add_subplot(gs[1, 3])
        panel_label_o2 = "g)"

    if spk_z_t.size > 0:
        ax8.plot(spk_z_t, spk_z_i, "|", markersize=2.0)
    ax8.axvline(cfg.cue_end, linestyle="--", linewidth=0.9, color="k", alpha=0.8)
    set_panel_title(ax8, panel_label_o2, r"Spiking activity $o_2$", pad=12)
    ax8.set_xlabel("Time, s")
    ax8.tick_params(axis="both", labelsize=16)
    ax8.set_ylabel(r"Neurons $o_2$")
    ax8.set_ylim(0, cfg.n_lat)
    ax8.grid(True, linestyle="--", alpha=0.35)

    # Не используем tight_layout: для 3D-осей он часто создаёт слишком большие пустые поля.

    fig.subplots_adjust(
        left=0.060,   # больше отступ слева
        right=0.985,  # больше отступ справа
        bottom=0.080, # больше отступ снизу
        top=0.890,    # больше отступ сверху
        wspace=0.3,
        hspace=0.2,
    )
    # fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.35)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_components(result, cfg: Config, out_path):
    t = result["t"]
    ref = result["ref"]
    u = result["u"]
    g = result["g"]
    N_sens = result["N_sens"]
    arch = result["arch"]
    signal_title = result["signal_title"]
    labels = result["labels"]
    D = ref.shape[1]
    i0 = int(np.searchsorted(t, cfg.plot_from_sec))

    fig, axes = plt.subplots(D, 1, figsize=(18, 4.0 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.plot(t[i0:], ref[i0:, d], linewidth=1.2, label="reference")
        ax.plot(t[i0:], u[i0:, d], linewidth=1.2, label=r"autonomous $o_1$")
        ax.plot(t[i0:], g[i0:, d], linewidth=1.1, linestyle=(0, (5, 10)), label=r"$g$")
        ax.axvline(cfg.cue_end, linestyle="--", linewidth=1.0)
        ax.set_ylabel(labels[d])
        ax.set_title(f"Component {labels[d]}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False, loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time, s")
    fig.suptitle(
        rf"{signal_title} | {arch}, $N_{{\mathrm{{sens}}}} = {N_sens}$"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_summary_for_signal(df_sig, cfg: Config, out_dir: Path, signal_name: str):
    metrics = [
        ("rmse_o1_g", "RMSE(o1, g)", False),
        ("mean_amp_corr", "Mean amplitude correlation", True),
        ("mean_freq_corr", "Mean frequency correlation", True),
        ("mean_plv", "Mean PLV", True),
        ("sim_time_s", "Simulation time, s", False),
        ("sim_python_peak_mem_mb", "Python peak memory, MB", False),
        ("sim_rss_peak_delta_mb", "RSS peak delta, MB", False),
    ]

    for col, ylabel, ylim01 in metrics:
        if col not in df_sig.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for arch, g in df_sig.groupby("arch"):
            g = g.sort_values("N_sens")
            ax.plot(g["N_sens"], g[col], marker="o", linewidth=1.8, label=arch)
        ax.set_xlabel(r"Number of sensory neurons $N_{sens}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{signal_name}: {ylabel}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        if ylim01:
            ax.set_ylim(-0.05, 1)
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_{signal_name}_{col}.png", dpi=cfg.dpi, bbox_inches="tight")
        if cfg.show_plots:
            plt.show()
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel_cols = ["rmse_o1_g", "mean_amp_corr", "mean_freq_corr", "mean_plv"]
    panel_titles = ["RMSE(o1, g)", "Amp. corr.", "Freq. corr.", "PLV"]
    for ax, col, title in zip(axes.flat, panel_cols, panel_titles):
        for arch, g in df_sig.groupby("arch"):
            g = g.sort_values("N_sens")
            ax.plot(g["N_sens"], g[col], marker="o", linewidth=1.8, label=arch)
        ax.set_title(title)
        ax.set_xlabel(r"$N_{sens}$")
        ax.grid(True, linestyle="--", alpha=0.35)
        if col != "rmse_o1_g":
            ax.set_ylim(-0.05, 1)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_{signal_name}_quality_panel.png", dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_summary(df, cfg: Config, out_dir: Path):
    for signal_name, df_sig in df.groupby("signal"):
        plot_summary_for_signal(df_sig, cfg, out_dir, signal_name)


# =============================================================================
# Training recurrent sensory decoder and running architectures
# =============================================================================

def make_train_indices(N_t, max_train_samples):
    if N_t <= max_train_samples:
        return np.arange(N_t, dtype=int)
    idx = np.linspace(0, N_t - 1, max_train_samples, dtype=int)
    return np.unique(idx)


def train_sensory_recurrent_decoder(ref, rec_target, enc_s, gain_s, bias_s, cfg: Config):
    N_t = ref.shape[0]
    N_sens = enc_s.shape[0]
    train_idx = make_train_indices(N_t, cfg.max_train_samples)
    A_train = np.zeros((train_idx.size, N_sens), dtype=np.float32)
    Y_train = rec_target[train_idx].astype(np.float32)

    V = np.zeros(N_sens, dtype=float)
    ref_count = np.zeros(N_sens, dtype=int)
    a = np.zeros(N_sens, dtype=float)

    j = 0
    for n in range(N_t):
        V, ref_count, spikes = lif_population_step(
            V, ref_count, ref[n], enc_s, gain_s, bias_s,
            cfg.dt, cfg.tau_rc, cfg.tau_ref,
        )
        a = update_filtered_activity(a, spikes, cfg.dt, cfg.tau_syn)
        if j < train_idx.size and n == train_idx[j]:
            A_train[j] = a.astype(np.float32)
            j += 1

    W_s_rec = solve_ridge(A_train, Y_train, cfg.ridge_lambda_sens)
    return W_s_rec





def run_architecture(
    arch,
    signal_info,
    W_s_rec,
    enc_s,
    gain_s,
    bias_s,
    cfg: Config,
    seed,
):
    arch = arch.upper()
    if arch not in {"PC-EC", "PC-SC"}:
        raise ValueError("arch must be 'PC-EC' or 'PC-SC'")

    rng = np.random.default_rng(seed)
    ref = signal_info["ref"]
    t = signal_info["t"]
    labels = signal_info["labels"]
    D = ref.shape[1]
    N_t = t.size
    N_sens = enc_s.shape[0]
    use_error_population = arch == "PC-SC"

    enc_z, gain_z, bias_z = init_population(
        cfg.n_lat, D, rng, cfg.tau_rc, cfg.tau_ref, cfg.rate_low, cfg.rate_high
    )

    if use_error_population:
        # PC-SC: this is the real spiking error population used for e_dec and z update.
        enc_e, gain_e, bias_e = init_population(
            cfg.n_err, D, rng, cfg.tau_rc, cfg.tau_ref, cfg.rate_low, cfg.rate_high
        )
        W_e = solve_static_decoder(enc_e, gain_e, bias_e, D, rng, cfg, target="identity")
    else:
        # PC-EC: no spiking error population is created.
        # The latent state z is driven directly by the continuous error e.
        enc_e = gain_e = bias_e = W_e = None

    u = np.ones((N_t, D), dtype=float)
    z = np.zeros((N_t, D), dtype=float)
    g = np.zeros((N_t, D), dtype=float)
    e_true = np.zeros((N_t, D), dtype=float)

    g_hat = np.zeros(D, dtype=float)

    V_s = np.zeros(N_sens, dtype=float)
    ref_s = np.zeros(N_sens, dtype=int)
    a_s = np.zeros(N_sens, dtype=float)

    V_z = np.zeros(cfg.n_lat, dtype=float)
    ref_z = np.zeros(cfg.n_lat, dtype=int)
    a_z = np.zeros(cfg.n_lat, dtype=float)

    if use_error_population:
        V_e = np.zeros(cfg.n_err, dtype=float)
        ref_e = np.zeros(cfg.n_err, dtype=int)
        a_e = np.zeros(cfg.n_err, dtype=float)
    else:
        V_e = ref_e = a_e = None

    W_pred = np.zeros((cfg.n_lat, D), dtype=float)

    spike_times_s, spike_ids_s = [], []
    spike_times_z, spike_ids_z = [], []
    spike_times_e, spike_ids_e = [], []

    for n in range(1, N_t):
        # 1) Autonomous sensory population o1
        V_s, ref_s, spikes_s = lif_population_step(
            V_s, ref_s, u[n - 1], enc_s, gain_s, bias_s,
            cfg.dt, cfg.tau_rc, cfg.tau_ref,
        )
        append_spikes(
            spike_times_s, spike_ids_s, t[n], spikes_s,
            max_neurons=cfg.spike_plot_max_neurons,
        )
        a_s = update_filtered_activity(a_s, spikes_s, cfg.dt, cfg.tau_syn)
        rec_s = a_s @ W_s_rec

        if t[n] < cfg.cue_end:
            cue = ref[n - 1] - u[n - 1]
        else:
            cue = np.zeros(D)

        u_dot = (-u[n - 1] + rec_s + cue) / cfg.tau_o1
        u[n] = u[n - 1] + cfg.dt * u_dot


        # 2) Prediction error
        e = u[n] - g_hat
        e_true[n] = e

        # 3) Error pathway into latent state z
        if use_error_population:
            # PC-SC: the explicit spiking error population is decoded as e_dec and drives z.
            V_e, ref_e, spikes_e = lif_population_step(
                V_e, ref_e, np.clip(e, -1.0, 1.0), enc_e, gain_e, bias_e,
                cfg.dt, cfg.tau_rc, cfg.tau_ref,
            )
            append_spikes(spike_times_e, spike_ids_e, t[n], spikes_e, max_neurons=cfg.n_err)
            a_e = update_filtered_activity(a_e, spikes_e, cfg.dt, cfg.tau_syn)
            e_drive = a_e @ W_e
        else:
            # PC-EC: direct continuous error, no error spikes and no monitor population.
            e_drive = e

        z_dot = (-z[n - 1] / cfg.tau_z) + cfg.k_error_to_z * e_drive
        z[n] = z[n - 1] + cfg.dt * z_dot
        z[n] = np.clip(z[n], -1.0, 1.0)

        # 4) Latent population z and top-down prediction decoder
        V_z, ref_z, spikes_z = lif_population_step(
            V_z, ref_z, z[n], enc_z, gain_z, bias_z,
            cfg.dt, cfg.tau_rc, cfg.tau_ref,
        )
        append_spikes(spike_times_z, spike_ids_z, t[n], spikes_z, max_neurons=cfg.n_lat)
        a_z = update_filtered_activity(a_z, spikes_z, cfg.dt, cfg.tau_syn)

        # g_hat = a_z @ W_pred
        tau_g = 0.02
        g_raw = a_z @ W_pred
        g_hat = g_hat + (cfg.dt / tau_g) * (g_raw - g_hat)
        g[n] = g_hat

        # 5) PES-like local update for top-down decoder
        W_pred += cfg.eta * np.outer(a_z, e) * cfg.dt

    spk_s = concat_spikes(spike_times_s, spike_ids_s)
    spk_z = concat_spikes(spike_times_z, spike_ids_z)
    spk_e = concat_spikes(spike_times_e, spike_ids_e)

    metrics = summarize_sync(u, g, ref, t, cfg, labels)

    return {
        "signal": signal_info["signal"],
        "signal_title": signal_info["title"],
        "labels": labels,
        "arch": arch,
        "N_sens": N_sens,
        "D": D,
        "t": t,
        "ref": ref,
        "u": u,
        "z": z,
        "g": g,
        "e_true": e_true,
        "spikes_o1": spk_s,
        "spikes_z": spk_z,
        "spikes_error": spk_e,
        "metrics": metrics,
    }


# =============================================================================
# Main experiment
# =============================================================================

def run_sweep(cfg: Config):
    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")

    rows = []

    print("\n=== PC-SC vs PC-EC sweep: Lorenz + oscillator ===")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Signals: {cfg.signal_names}")
    print(f"N_sens values: {cfg.n_sens_values}\n")

    for signal_name in cfg.signal_names:
        signal_info = make_reference_signal(signal_name, cfg)
        ref = signal_info["ref"]
        t = signal_info["t"]
        D = signal_info["D"]
        labels = signal_info["labels"]

        dref_dt = np.gradient(ref, cfg.dt, axis=0)
        rec_target = ref + cfg.tau_o1 * dref_dt

        print(f"\n################ {signal_info['title']} | D={D} ################")

        for N_sens in cfg.n_sens_values:
            print(f"\n================ {signal_name}, N_sens={N_sens} ================")
            rng_s = np.random.default_rng(cfg.base_seed + 100000 + N_sens + 999 * D)
            enc_s, gain_s, bias_s = init_population(
                N_sens, D, rng_s, cfg.tau_rc, cfg.tau_ref,
                cfg.rate_low, cfg.rate_high,
            )

            W_s_rec, train_stats = measure_call(
                train_sensory_recurrent_decoder,
                ref, rec_target, enc_s, gain_s, bias_s, cfg,
            )
            print(
                f"Sensory decoder trained: "
                f"time={train_stats['time_s']:.2f}s, "
                f"py_peak={train_stats['python_peak_mem_mb']:.1f}MB, "
                f"rss_peak_delta={train_stats['rss_peak_delta_mb']:.1f}MB"
            )

            for arch_idx, arch in enumerate(["PC-EC", "PC-SC"]):
                seed = cfg.base_seed + 10_000 * arch_idx + N_sens + 999 * D
                result, sim_stats = measure_call(
                    run_architecture,
                    arch, signal_info, W_s_rec, enc_s, gain_s, bias_s, cfg, seed,
                )

                m = result["metrics"]
                row = {
                    "signal": signal_info["signal"],
                    "signal_title": signal_info["title"],
                    "D": D,
                    "N_sens": N_sens,
                    "arch": arch,
                    "train_sens_time_s": train_stats["time_s"],
                    "train_sens_python_peak_mem_mb": train_stats["python_peak_mem_mb"],
                    "train_sens_rss_peak_delta_mb": train_stats["rss_peak_delta_mb"],
                    "train_sens_rss_delta_mb": train_stats["rss_delta_mb"],
                    "sim_time_s": sim_stats["time_s"],
                    "sim_python_peak_mem_mb": sim_stats["python_peak_mem_mb"],
                    "sim_rss_peak_delta_mb": sim_stats["rss_peak_delta_mb"],
                    "sim_rss_delta_mb": sim_stats["rss_delta_mb"],
                    **m,
                }
                rows.append(row)

                print(
                    f"{arch}: RMSE(ref, o1)={m['rmse_ref_o1']:.6f}, "
                    f"{arch}: RMSE(o1,g)={m['rmse_o1_g']:.6f}, "
                    f"amp={m['mean_amp_corr']:.3f}, "
                    f"freq={m['mean_freq_corr']:.3f}, "
                    f"PLV={m['mean_plv']:.3f}, "
                    f"sim_time={sim_stats['time_s']:.2f}s, "
                    f"py_peak={sim_stats['python_peak_mem_mb']:.1f}MB, "
                    f"rss_peak_delta={sim_stats['rss_peak_delta_mb']:.1f}MB"
                )

                if cfg.save_each_n_figures:
                    safe_arch = arch.replace("-", "_")
                    safe_signal = signal_info["signal"]
                    plot_overview(
                        result, cfg,
                        fig_dir / f"overview_{safe_signal}_{safe_arch}_N{N_sens:04d}.png",
                    )
                    plot_components(
                        result, cfg,
                        fig_dir / f"components_{safe_signal}_{safe_arch}_N{N_sens:04d}.pdf",
                    )

                del result
                gc.collect()

            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(out_dir / "summary_metrics_partial.csv", index=False)

            del W_s_rec, enc_s, gain_s, bias_s
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_metrics.csv", index=False, sep=';', decimal=',')

    print("\n=== Final table ===")
    cols_to_print = [
        "signal", "N_sens", "arch", "rmse_o1_g", "mean_amp_corr",
        "mean_freq_corr", "mean_plv", "sim_time_s",
        "sim_python_peak_mem_mb", "sim_rss_peak_delta_mb",
    ]
    print(df[cols_to_print].to_string(index=False))

    if cfg.save_comparison_figures:
        plot_summary(df, cfg, fig_dir)

    print(f"\nSaved CSV: {out_dir / 'summary_metrics.csv'}")
    print(f"Saved figures: {fig_dir}")
    return df


if __name__ == "__main__":
    # Быстрый тест перед полной серией:
    # CFG.signal_names = ("oscillator",)
    # CFG.n_sens_values = (100, 200)
    # CFG.T = 10.0
    # CFG.cue_end = 3.0
    # CFG.max_train_samples = 3000
    # CFG.decoder_train_samples = 1000
    # CFG.dpi = 120

    # Только Lorenz:
    # CFG.signal_names = ("lorenz",)

    # Только oscillator:
    # CFG.signal_names = ("oscillator",)

    run_sweep(CFG)