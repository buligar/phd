"""
Hyperparameter ablation (one-at-a-time sensitivity analysis) for the PC-SC vs PC-EC
predictive-coding spiking networks -- SINGLE-FIGURE RMSE VERSION.

WHAT THIS PRODUCES
------------------
One figure per reference signal, with 7 x 2 = 14 panels:

    rows    = the seven ablated hyperparameters
              (eta, tau_syn, tau_rc, tau_ref,
               ridge_lambda_sens, ridge_lambda_dec, cue_end)
    columns = the two RMSE metrics listed in AblationConfig.rmse_metrics
              (default: RMSE(o1, g) and RMSE(reference, g))

Every panel shows PC-EC and PC-SC (mean +- std over seeds) and a dashed vertical
line at the baseline (paper) value of that parameter, so a reader immediately
sees whether the reported performance sits on a plateau or on a slope.

WHAT CHANGED vs the previous version
------------------------------------
* The per-parameter 8-panel figures are replaced by the single grid above.
  All other metrics (amplitude/frequency correlation, PLV, phase, convergence
  times, runtime, memory) are STILL computed and written to the CSVs -- only
  the plotting was reduced.
* rmse_ref_o1 is deliberately NOT a default column: the autonomous sensory
  trajectory u does not depend on the architecture (see run_architecture: u is
  driven only by rec_s and the cue), so PC-EC and PC-SC would overlap exactly.
  It is available via AblationConfig.rmse_metrics if you want it as a check.
* Baseline de-duplication: the baseline value of every parameter produces an
  identical Config, so it was previously simulated 7 times. Results are now
  cached per (config, seed, arch), which removes ~20% of the runs.
* cue_end sweep now uses a COMMON evaluation window across values
  (align_eval_window=True). Previously the metric window started at
  cue_end + sync_start_offset_after_cue, i.e. a different window per value.
  Set align_eval_window=False to reproduce the old behaviour.
* conv_reached_threshold is now aggregated as a convergence fraction.
* Fixed: QUICK_TEST used signal_names=("oscillator") -- a string, not a tuple.

HOW TO USE
----------
1. Put this file next to your main simulation script and set SIM_MODULE below
   to that file's name (without the .py extension).
2. Edit ABLATION_GRID / AblationConfig to choose parameters, ranges,
   population size, number of seeds, and which reference signals to run.
3. Run:  python hyperparameter_ablation.py
   For a fast end-to-end check first, set QUICK_TEST = True.

OUTPUTS (written to AblationConfig.out_dir)
-------------------------------------------
* ablation_raw.csv                    one row per (signal, param, value, arch, seed)
* ablation_aggregated.csv             mean/std over seeds
* figures/ablation_rmse_grid_<signal>.png / .pdf    the 14-panel figure
"""

import os
# Headless backend must be selected before matplotlib.pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

import gc
import sys
import importlib
import warnings
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   # <-- put this with the other imports

# One place to control the size of every tick label in the figure.
TICK_LABELSIZE = 13

# =============================================================================
# 0. Import the simulation building blocks from your main script
# =============================================================================

# <-- SET THIS to the filename (without ".py") of your main simulation file.
SIM_MODULE = "PC-EC_PC-SC"

# Make a sibling file importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_sim = importlib.import_module(SIM_MODULE)

Config = _sim.Config
make_reference_signal = _sim.make_reference_signal
init_population = _sim.init_population
train_sensory_recurrent_decoder = _sim.train_sensory_recurrent_decoder
run_architecture = _sim.run_architecture
measure_call = _sim.measure_call


# =============================================================================
# 1. What to ablate
# =============================================================================

# Each entry: parameter name -> list of values to test.
ABLATION_GRID = {
    "eta":               [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   # PES learning rate
    "tau_syn":           [0.01, 0.02, 0.05, 0.10, 0.20],   # synaptic time constant
    "tau_rc":            [0.005, 0.01, 0.02, 0.05],        # membrane time constant
    "tau_ref":           [0.001, 0.002, 0.003, 0.004],     # refractory period
    "ridge_lambda_sens": [1e-3, 1e-2, 1e-1],               # sensory decoder regularization
    "ridge_lambda_dec":  [1e-3, 1e-2, 1e-1],               # error decoder regularization (PC-SC)
    "cue_end":           [1, 2, 3, 4, 5],                  # forcing (cue) duration
}
# ABLATION_GRID = {
#     "eta":               [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   # PES learning rate
#     "tau_syn":           [0.01, 0.02, 0.05, 0.10, 0.20],   # synaptic time constant
# }
# Row order of the final figure. Only parameters present in the sweep are drawn.
PARAM_ORDER = (
    "eta", "tau_syn", "tau_rc", "tau_ref",
    "ridge_lambda_sens", "ridge_lambda_dec", "cue_end",
)
# PARAM_ORDER = (
#     "eta", "tau_syn",
# )

# Some "parameters" map to more than one Config field.
# In the manuscript a single symbol tau_syn is used for the synaptic filter AND
# for the sensory/latent state dynamics. The code splits these into
# tau_syn / tau_o1 / tau_z. To stay faithful to the paper, sweeping "tau_syn"
# sets all three together. Everything else maps to the field of the same name.
PARAM_TO_FIELDS = {
    "tau_syn": ["tau_syn", "tau_o1", "tau_z"],
}

# Parameters that span orders of magnitude are plotted on a log x-axis.
LOG_PARAMS = {"eta", "tau_syn", "tau_rc", "tau_ref",
              "ridge_lambda_sens", "ridge_lambda_dec"}

# Axis labels used in the figure.
PARAM_LABEL = {
    "eta":               r"$\eta$",
    "tau_syn":           r"$\tau_{\mathrm{syn}}$, s",
    "tau_rc":            r"$\tau_{\mathrm{RC}}$, s",
    "tau_ref":           r"$\tau_{\mathrm{ref}}$, s",
    "ridge_lambda_sens": r"$\lambda_{\mathrm{sens}}$",
    "ridge_lambda_dec":  r"$\lambda_{\mathrm{dec}}$",
    "cue_end":           r"cue duration, s",
}

RMSE_LABEL = {
    "rmse_o1_g":   r"RMSE($o_1$, $g$)",
    "rmse_ref_g":  r"RMSE(ref, $g$)",
    "rmse_ref_o1": r"RMSE(ref, $o_1$)",
}

_ARCH_STYLE = {
    "PC-EC": dict(color="tab:blue",   marker="o", linestyle="-"),
    "PC-SC": dict(color="tab:orange", marker="s", linestyle="-"),
}



@dataclass
class AblationConfig:
    # Which reference dynamics to analyse. One figure is produced per signal.
    signal_names: tuple = ("oscillator",)

    # A single, representative population size (held fixed while hyperparameters
    # vary). 540 matches the operating point of the main sweep.
    N_sens: int = 20

    # Random seeds per configuration. 10 matches the paper; raise to 20-30 for
    # tighter error bars in the final version (this also speaks to Reviewer 1.7).
    n_seeds: int = 10

    # Restrict to a subset of ABLATION_GRID keys, or None to use all of them.
    params: tuple = None

    # ---- figure -------------------------------------------------------------
    # The two RMSE columns of the 14-panel figure. Order = left, right.
    rmse_metrics: tuple = ("rmse_ref_o1", "rmse_o1_g")
    panel_w: float = 6.2            # width  of one panel, inches
    panel_h: float = 2.9            # height of one panel, inches
    log_y: str = "auto"             # "auto" | "always" | "never"
    log_y_dynamic_range: float = 30.0   # switch to log y when max/min exceeds this
    share_y_per_column: bool = False    # True -> identical y-limits down a column
    save_pdf: bool = True

    # ---- evaluation window --------------------------------------------------
    # Metrics start at cue_end + sync_start_offset_after_cue. When cue_end is
    # swept this moves the window, so values are not compared on the same data.
    # align_eval_window keeps the window start fixed at eval_start_s instead.
    align_eval_window: bool = True
    eval_start_s: float = None      # None -> base cue_end + base offset

    # ---- convergence-speed metric (see convergence_metrics()) ---------------
    conv_win_sec: float = 1.0       # trailing window for the RMS error curve
    conv_tol: float = 0.20          # settling band = (1+tol) * steady-state error
    conv_steady_sec: float = 10.0   # tail length used to estimate steady-state error
    conv_abs_threshold: float = 0.05  # "time to reach and stay below this RMSE"

    out_dir: str = "results_ablation"
    dpi: int = 200


# =============================================================================
# 2. Convergence-speed metric (computed from the returned trajectories)
# =============================================================================

def _trailing_rms(x, win):
    """Causal (trailing) root-mean-square of x over a window of `win` samples.

    Uses an expanding window for the first `win-1` points so the very start is
    not biased downward.
    """
    x2 = np.asarray(x, dtype=float) ** 2
    c = np.cumsum(x2)
    idx = np.arange(x2.size)
    lo = np.maximum(idx - win + 1, 0)
    csum = c[idx] - np.where(lo > 0, c[lo - 1], 0.0)
    counts = (idx - lo + 1).astype(float)
    return np.sqrt(csum / counts)


def convergence_metrics(u, g, t, dt, cfg_ab: AblationConfig):
    """Quantify how fast the top-down prediction g locks onto o1.

    Returns:
        conv_threshold_time_s : first time the trailing-RMS o1-g error drops
                                below cfg_ab.conv_abs_threshold and stays there
                                (NaN if that accuracy is never reached -> a clear
                                "did not converge to this level" signal).
        conv_settle_time_s    : classic settling time relative to the run's own
                                steady-state error (always defined).
        conv_reached_threshold: bool, whether the absolute threshold was met.
        steady_err_o1_g       : steady-state RMS o1-g error over the last
                                conv_steady_sec seconds.
    """
    err = np.linalg.norm(np.asarray(u) - np.asarray(g), axis=1)
    win = max(1, int(round(cfg_ab.conv_win_sec / dt)))
    curve = _trailing_rms(err, win)

    n_steady = max(1, int(round(cfg_ab.conv_steady_sec / dt)))
    n_steady = min(n_steady, err.size)
    steady = float(np.sqrt(np.mean(err[-n_steady:] ** 2)))

    # Settling time relative to steady state: last time above the band, + 1.
    band = (1.0 + cfg_ab.conv_tol) * steady
    above = curve > band
    if np.any(above):
        settle_idx = min(int(np.max(np.flatnonzero(above))) + 1, t.size - 1)
    else:
        settle_idx = 0
    t_settle = float(t[settle_idx])

    # Absolute threshold: last time above threshold, + 1 (NaN if never below).
    thr = cfg_ab.conv_abs_threshold
    reached = False
    t_thresh = np.nan
    if thr is not None:
        not_below = curve > thr
        k = (int(np.max(np.flatnonzero(not_below))) + 1) if np.any(not_below) else 0
        if k < t.size:
            t_thresh = float(t[k])
            reached = True

    return {
        "conv_threshold_time_s": t_thresh,
        "conv_settle_time_s": t_settle,
        "conv_reached_threshold": reached,
        "steady_err_o1_g": steady,
    }


def _mean_abs_phase_deg(metrics, labels):
    """Average the per-coordinate |mean phase difference| into one number."""
    vals = [abs(metrics.get(f"mean_phase_deg_{lab}", np.nan)) for lab in labels]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# =============================================================================
# 3. Caches
#    (a) sensory decoder: not retrained when the swept parameter cannot change it
#    (b) whole runs: the baseline value of every parameter yields an identical
#        Config, so it is simulated once instead of seven times
# =============================================================================

def _sensory_key(cfg, signal_info, N_sens, sens_seed):
    # Every field that changes the trained sensory encoder/decoder. If none of
    # these change between two runs, the decoder can be reused.
    return (
        signal_info["signal"], signal_info["D"], N_sens, sens_seed,
        cfg.tau_rc, cfg.tau_ref, cfg.tau_syn, cfg.tau_o1,
        cfg.ridge_lambda_sens, cfg.max_train_samples,
        cfg.rate_low, cfg.rate_high, cfg.T, cfg.dt,
    )


def _get_sensory_decoder(cfg, signal_info, N_sens, sens_seed, cache):
    key = _sensory_key(cfg, signal_info, N_sens, sens_seed)
    if key in cache:
        return cache[key]

    ref = signal_info["ref"]
    dref_dt = np.gradient(ref, cfg.dt, axis=0)
    rec_target = ref + cfg.tau_o1 * dref_dt

    rng_s = np.random.default_rng(sens_seed)
    enc_s, gain_s, bias_s = init_population(
        N_sens, signal_info["D"], rng_s,
        cfg.tau_rc, cfg.tau_ref, cfg.rate_low, cfg.rate_high,
    )
    W_s_rec, train_stats = measure_call(
        train_sensory_recurrent_decoder,
        ref, rec_target, enc_s, gain_s, bias_s, cfg,
    )
    cache[key] = (enc_s, gain_s, bias_s, W_s_rec, train_stats)
    return cache[key]


def _cfg_key(cfg):
    """Hashable snapshot of a Config (all fields are scalars or tuples)."""
    return tuple(sorted(asdict(cfg).items()))


def _run_key(cfg, signal_name, N_sens, arch, seed):
    return (signal_name, N_sens, arch, seed, _cfg_key(cfg))


# =============================================================================
# 4. Evaluate one parameter value (all seeds, both architectures)
# =============================================================================

def _resolve_overrides(param, value, base_cfg, ab: AblationConfig):
    """Config fields to change for this (param, value), incl. window alignment."""
    overrides = {f: value for f in PARAM_TO_FIELDS.get(param, [param])}

    if param == "cue_end" and ab.align_eval_window:
        eval_start = ab.eval_start_s
        if eval_start is None:
            eval_start = base_cfg.cue_end + base_cfg.sync_start_offset_after_cue
        offset = float(eval_start) - float(value)
        if offset < 0.0:
            warnings.warn(
                f"align_eval_window: cue_end={value} is past the requested "
                f"evaluation start {eval_start}s; keeping the default offset."
            )
        else:
            overrides["sync_start_offset_after_cue"] = offset
    return overrides


def _evaluate_value(param, value, base_cfg, ab: AblationConfig,
                    signal_info, sensory_cache, run_cache):
    cfg = replace(base_cfg, **_resolve_overrides(param, value, base_cfg, ab))
    D = signal_info["D"]
    labels = signal_info["labels"]
    signal_name = signal_info["signal"]
    rows = []

    for s in range(ab.n_seeds):
        # Reproducible per-trial seeds, mirroring the offsets used in run_sweep.
        trial_seed = base_cfg.base_seed + 7919 * s
        sens_seed = trial_seed + 100000 + ab.N_sens + 999 * D

        for arch_idx, arch in enumerate(("PC-EC", "PC-SC")):
            seed = trial_seed + 10_000 * arch_idx + ab.N_sens + 999 * D
            key = _run_key(cfg, signal_name, ab.N_sens, arch, seed)

            if key in run_cache:
                payload = run_cache[key]
            else:
                enc_s, gain_s, bias_s, W_s_rec, train_stats = _get_sensory_decoder(
                    cfg, signal_info, ab.N_sens, sens_seed, sensory_cache
                )
                result, sim_stats = measure_call(
                    run_architecture,
                    arch, signal_info, W_s_rec, enc_s, gain_s, bias_s, cfg, seed,
                )
                m = result["metrics"]
                conv = convergence_metrics(
                    result["u"], result["g"], result["t"], cfg.dt, ab
                )
                payload = {
                    # synchronization accuracy
                    "mean_amp_corr": m.get("mean_amp_corr", np.nan),
                    "mean_freq_corr": m.get("mean_freq_corr", np.nan),
                    "mean_plv": m.get("mean_plv", np.nan),
                    "mean_phase_deg": _mean_abs_phase_deg(m, labels),
                    # reconstruction error
                    "rmse_o1_g": m.get("rmse_o1_g", np.nan),
                    "rmse_ref_o1": m.get("rmse_ref_o1", np.nan),
                    "rmse_ref_g": m.get("rmse_ref_g", np.nan),
                    # convergence speed
                    "conv_threshold_time_s": conv["conv_threshold_time_s"],
                    "conv_settle_time_s": conv["conv_settle_time_s"],
                    "conv_reached_threshold": conv["conv_reached_threshold"],
                    "steady_err_o1_g": conv["steady_err_o1_g"],
                    # cost
                    "sim_time_s": sim_stats["time_s"],
                    "sim_python_peak_mem_mb": sim_stats["python_peak_mem_mb"],
                    "sim_rss_peak_delta_mb": sim_stats["rss_peak_delta_mb"],
                    "train_time_s": train_stats["time_s"],
                }
                run_cache[key] = payload

                del result
                gc.collect()

            rows.append({
                "signal": signal_name,
                "param": param,
                "value": value,
                "arch": arch,
                "seed": s,
                "N_sens": ab.N_sens,
                **payload,
            })

    return rows


# =============================================================================
# 5. Aggregation
# =============================================================================

_AGG_METRICS = [
    "rmse_o1_g", "rmse_ref_g", "rmse_ref_o1",
    "mean_amp_corr", "mean_freq_corr", "mean_plv", "mean_phase_deg",
    "conv_threshold_time_s", "conv_settle_time_s", "conv_reached_threshold",
    "steady_err_o1_g",
    "sim_time_s", "sim_python_peak_mem_mb", "sim_rss_peak_delta_mb",
]


def aggregate(df):
    metrics = [m for m in _AGG_METRICS if m in df.columns]
    df = df.copy()
    if "conv_reached_threshold" in df.columns:
        # mean of the bool -> fraction of seeds that reached the threshold
        df["conv_reached_threshold"] = df["conv_reached_threshold"].astype(float)
    keys = ["signal", "param", "value", "arch"]
    agg = df.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


# =============================================================================
# 6. The single 14-panel figure
# =============================================================================

def _panel_uses_log_y(values, ab: AblationConfig):
    if ab.log_y == "never":
        return False
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0 or np.any(v <= 0):
        return False
    if ab.log_y == "always":
        return True
    return float(np.max(v) / np.min(v)) > ab.log_y_dynamic_range


# =============================================================================
# 6. The single 14-panel figure   -- DROP-IN REPLACEMENT for section 6
#    Add `import matplotlib.ticker as mticker` at the top of the file.
# =============================================================================



def _pow10_label(v, _pos=None):
    r"""Compact LaTeX tick label: $10^{k}$ or $m\times 10^{k}$."""
    if not np.isfinite(v) or v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = round(v / 10.0 ** exp, 3)
    if abs(mant) == 1.0:
        return rf"$10^{{{exp}}}$" if mant > 0 else rf"$-10^{{{exp}}}$"
    return rf"${mant:g}\times 10^{{{exp}}}$"


def _set_value_ticks(ax, values, log):
    """Tick the x-axis exactly at the swept parameter values.

    Must be called AFTER ax.set_xscale(), because changing the scale resets
    locators and formatters.
    """
    v = np.unique(np.asarray(list(values), dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    ax.xaxis.set_major_locator(mticker.FixedLocator(v))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(_pow10_label if log else lambda x, p: f"{x:g}")
    )
    # No minor ticks -> no second, differently sized set of labels.
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _tidy_log_yaxis(ax, values):
    """Make sure a log y-axis spanning ~1 decade still gets several labels."""
    v = np.asarray([y for y in values if np.isfinite(y) and y > 0], dtype=float)
    if v.size == 0:
        return
    if np.log10(v.max() / v.min()) <= 1.5:
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0))
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pow10_label))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _panel_uses_log_y(values, ab: AblationConfig):
    if ab.log_y == "never":
        return False
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0 or np.any(v <= 0):
        return False
    if ab.log_y == "always":
        return True
    return float(np.max(v) / np.min(v)) > ab.log_y_dynamic_range


def plot_rmse_grid(agg, signal, params, baseline, ab: AblationConfig, out_dir):
    """One figure: rows = parameters, columns = ab.rmse_metrics."""
    metrics = [m for m in ab.rmse_metrics
               if f"{m}_mean" in agg.columns]
    params = [p for p in params if not agg[(agg["signal"] == signal)
                                           & (agg["param"] == p)].empty]
    if not params or not metrics:
        print("plot_rmse_grid: nothing to plot.")
        return None

    nrows, ncols = len(params), len(metrics)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ab.panel_w * ncols, ab.panel_h * nrows),
        squeeze=False,
    )

    letters = [f"{chr(ord('a') + i)})" for i in range(nrows * ncols)]
    legend_handles, legend_labels = [], []
    col_limits = {j: [np.inf, -np.inf] for j in range(ncols)}

    for i, param in enumerate(params):
        sub_p = agg[(agg["signal"] == signal) & (agg["param"] == param)]
        is_log_x = param in LOG_PARAMS
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            mean_col, std_col = f"{metric}_mean", f"{metric}_std"

            all_y = []
            for arch, style in _ARCH_STYLE.items():
                d = sub_p[sub_p["arch"] == arch].sort_values("value")
                if d.empty:
                    continue
                x = d["value"].to_numpy(dtype=float)
                y = d[mean_col].to_numpy(dtype=float)
                e = d[std_col].fillna(0.0).to_numpy(dtype=float)
                all_y.extend(y[np.isfinite(y)].tolist())

                line = ax.errorbar(
                    x, y, yerr=e, capsize=3, linewidth=1.8,
                    markersize=5, label=arch, **style,
                )
                if arch not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(arch)

            # ---- x-axis: scale first, then ticks at the swept values --------
            if is_log_x:
                ax.set_xscale("log")
            _set_value_ticks(ax, sub_p["value"].unique(), log=is_log_x)

            # ---- y-axis ------------------------------------------------------
            if _panel_uses_log_y(all_y, ab):
                ax.set_yscale("log")
                _tidy_log_yaxis(ax, all_y)

            bl = baseline.get(param)
            if bl is not None:
                vline = ax.axvline(bl, ls="--", color="k", alpha=0.55,
                                   linewidth=1.2, label="baseline")
                if "baseline" not in legend_labels:
                    legend_handles.append(vline)
                    legend_labels.append("baseline")

            ax.set_xlabel(PARAM_LABEL.get(param, param))
            ax.set_ylabel(RMSE_LABEL.get(metric, metric))
            ax.set_title(letters[i * ncols + j], loc="left",
                         fontsize=13, fontweight="bold", pad=6)
            ax.grid(True, ls="--", alpha=0.35)
            # which="both": major AND minor labels share one size.
            ax.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)

            if all_y:
                col_limits[j][0] = min(col_limits[j][0], float(np.min(all_y)))
                col_limits[j][1] = max(col_limits[j][1], float(np.max(all_y)))

    if ab.share_y_per_column:
        for j in range(ncols):
            lo, hi = col_limits[j]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.05 * (hi - lo)
                for i in range(nrows):
                    if axes[i][j].get_yscale() != "log":
                        axes[i][j].set_ylim(lo - pad, hi + pad)

    fig.legend(legend_handles, legend_labels, loc="upper right",
               frameon=False, ncol=len(legend_labels), fontsize=12,
               bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    out_dir = Path(out_dir)
    png_path = out_dir / f"ablation_rmse_grid_{signal}.png"
    fig.savefig(png_path, dpi=ab.dpi, bbox_inches="tight")
    if ab.save_pdf:
        fig.savefig(out_dir / f"ablation_rmse_grid_{signal}.pdf",
                    bbox_inches="tight")
    plt.close(fig)
    return png_path


# =============================================================================
# 7. Driver
# =============================================================================

def run_ablation(base_cfg, ab: AblationConfig, grid=None):
    grid = ABLATION_GRID if grid is None else grid

    requested = list(grid.keys()) if ab.params is None else list(ab.params)
    # Keep the figure row order stable, then append anything not in PARAM_ORDER.
    params = [p for p in PARAM_ORDER if p in requested]
    params += [p for p in requested if p not in PARAM_ORDER]

    out = Path(ab.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Reference trajectories depend only on the (fixed) signal parameters, so
    # build them once. For Lorenz this avoids re-running RK4 on every trial.
    signal_infos = {name: make_reference_signal(name, base_cfg)
                    for name in ab.signal_names}

    # Baseline value of each parameter (for the dashed lines in the figure).
    baseline = {}
    for p in params:
        f0 = PARAM_TO_FIELDS.get(p, [p])[0]
        baseline[p] = getattr(base_cfg, f0, None)

    n_runs = len(ab.signal_names) * sum(len(grid[p]) for p in params) * ab.n_seeds * 2
    print("=== Hyperparameter ablation (single RMSE figure) ===")
    print(f"Signals      : {ab.signal_names}")
    print(f"Parameters   : {params}")
    print(f"RMSE columns : {ab.rmse_metrics}")
    print(f"N_sens       : {ab.N_sens}")
    print(f"Seeds        : {ab.n_seeds}")
    print(f"Architectures: PC-EC, PC-SC")
    print(f"Planned architecture runs: {n_runs} "
          f"(duplicated baselines are served from cache)\n")

    sensory_cache, run_cache = {}, {}
    rows = []
    done_values = 0
    total_values = len(ab.signal_names) * sum(len(grid[p]) for p in params)

    for signal_name in ab.signal_names:
        signal_info = signal_infos[signal_name]
        for param in params:
            for value in grid[param]:
                done_values += 1
                print(f"[{done_values}/{total_values}] "
                      f"{signal_name} | {param} = {value}")
                rows += _evaluate_value(
                    param, value, base_cfg, ab,
                    signal_info, sensory_cache, run_cache,
                )
                # Save after each value so a long run can be inspected/resumed.
                pd.DataFrame(rows).to_csv(out / "ablation_raw_partial.csv",
                                          index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_raw.csv", index=False)

    agg = aggregate(df)
    agg.to_csv(out / "ablation_aggregated.csv", index=False)

    for signal_name in ab.signal_names:
        path = plot_rmse_grid(agg, signal_name, params, baseline, ab, fig_dir)
        if path is not None:
            print(f"Saved figure        : {path}")

    print(f"\nSaved raw CSV       : {out / 'ablation_raw.csv'}")
    print(f"Saved aggregated CSV: {out / 'ablation_aggregated.csv'}")
    print(f"Simulated runs      : {len(run_cache)} unique "
          f"(of {len(df)} table rows)")
    return df, agg


# Set to True for a fast end-to-end check (tiny signal, few seeds, short sim).
QUICK_TEST = False


def main():
    if QUICK_TEST:
        base_cfg = replace(
            Config(),
            T=30.0, cue_end=1.0, sync_start_offset_after_cue=1.0,
            max_train_samples=3000, decoder_train_samples=1000,
        )
        ab = AblationConfig(
            signal_names=("oscillator",),   # note the comma: this must be a tuple
            N_sens=20, n_seeds=3,
            conv_steady_sec=3.0, conv_win_sec=0.5, conv_abs_threshold=0.1,
            out_dir="results_ablation_quicktest",
        )
        small_grid = {
            "eta":               [1e-4, 1e-3, 1e-2],
            "tau_syn":           [0.02, 0.05, 0.10],
            "tau_rc":            [0.01, 0.02, 0.05],
            "tau_ref":           [0.001, 0.002, 0.004],
            "ridge_lambda_sens": [1e-3, 1e-2, 1e-1],
            "ridge_lambda_dec":  [1e-3, 1e-2, 1e-1],
            "cue_end":           [1, 2, 3],
        }
        run_ablation(base_cfg, ab, grid=small_grid)
    else:
        # Full run. Uses the paper's parameters as the baseline (Config defaults)
        # and the ABLATION_GRID / AblationConfig defined above.
        base_cfg = Config()
        ab = AblationConfig()
        run_ablation(base_cfg, ab)


if __name__ == "__main__":
    main()