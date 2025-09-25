import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    # Крупные размеры шрифтов
    "font.size": 22,          # базовый
    "axes.titlesize": 28,     # заголовки подграфиков
    "axes.labelsize": 26,     # подписи осей
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 30,   # suptitle
})

# ------------------------------ вспомогательные функции ------------------------------
def lif_rate(J, tau_rc=0.02, tau_ref=0.002):
    """
    Возвращает частоту стационарного LIF (Гц) при постоянном токе J (безразмерный порог = 1).
    J: (N_pre,) → out: (N_pre,)
    """
    J = np.asarray(J)                                 # (100,)
    out = np.zeros_like(J, dtype=float)               # (100,)
    mask = J > 1.0                                    # (100,) bool
    if np.any(mask):
        JJ = J[mask]                                  # (~M,)
        denom = tau_ref - tau_rc * np.log1p(-1.0 / JJ)  # (~M,)
        out[mask] = 1.0 / denom                       # (~M,)
    return out                                        # (100,4000)

def current_from_rate(r, tau_rc=0.02, tau_ref=0.002):
    """r: (N_pre,) → ток J: (N_pre,)"""
    r = np.maximum(np.asarray(r, dtype=float), 1e-3)  # (100,)
    k = (1.0 / r - tau_ref) / tau_rc                  # (100,)
    k = np.maximum(k, 1e-9)                           # (100,)
    return 1.0 / (1.0 - np.exp(-k))                   # (100,)

def unit_norm_rows(X, eps=1e-12):
    """X: (N_pre,3) → X/||X|| по строкам: (N_pre,3)"""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)    # (100,1)
    nrm = np.maximum(nrm, eps)                        # (100,1)
    return X / nrm                                    # (100,3)

def sample_states_for_decoder(K, rng, radius_xy=1.0):
    """
    Возвращает обучающую выборку X: (K,3) с (x1,x2) на диске и s∈[0,1].
    """
    u = rng.random(K)                                 # (K,)
    r = radius_xy * np.sqrt(u)                        # (K,)
    phi = rng.uniform(0, 2*np.pi, size=K)             # (K,)
    x1 = r * np.cos(phi)                              # (K,)
    x2 = r * np.sin(phi)                              # (K,)
    s  = rng.uniform(0.0, 1.0, size=K)                # (K,)
    X  = np.stack([x1, x2, s], axis=1)                # (K,3)
    return X                                          # (K,3)

# ------------------------------ параметры модели ------------------------------
rng = np.random.default_rng(1)

T, dt = 10.0, 0.001
t = np.arange(0.0, T + dt, dt)                        # (10001,)
N_t = t.size

omega = 10.0                                          # скаляр
tau_syn = 0.1                                         # скаляр
tau_rc  = 0.02                                        # скаляр
tau_ref = 0.002                                       # скаляр

N_pre = 100                                           # скаляр
reg   = 1e-2                                          # скаляр
eta   = 1e-3                                          # скаляр

# ------------------------------ источник u(t) = (x1,x2,speed) ------------------------------
theta = omega * dt                                    # скаляр
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])       # (2,2)
u = np.zeros((N_t, 3))                                # (10001,3)
u[0] = [1.0, 0.0, 1.0]                                # (3,)
for n in range(1, N_t):
    u[n, :2] = R @ u[n-1, :2]                         # (2,) = (2,2) @ (2,)
    u[n,  2] = 1.0                                    # скаляр

# ------------------------------ пред-ансамбль (NEF-параметры) ------------------------------
enc = unit_norm_rows(rng.normal(size=(N_pre, 3)))     # (100,3)
x_intercepts = rng.uniform(-0.95, 0.95, size=N_pre)   # (100,)
max_rates    = rng.uniform(100.0, 200.0, size=N_pre)  # (100,)
Jmax = current_from_rate(max_rates, tau_rc, tau_ref)  # (100,)
gain = (Jmax - 1.0) / (1.0 - x_intercepts)            # (100,)
bias = 1.0 - gain * x_intercepts                      # (100,)

# ------------------------------ начальные декодеры D0 (LS, тождественная функция) ------------------------------
K = 4000                                              # скаляр
X_samp = sample_states_for_decoder(K, rng, radius_xy=1.0)                # (K,3)
A_samp = lif_rate(enc @ X_samp.T * gain[:, None] + bias[:, None], tau_rc, tau_ref).T                                     # (K,100)
G = A_samp.T @ A_samp + reg * np.eye(N_pre)           # (100,100)
U = A_samp.T @ X_samp                                 # (100,3)
W = np.linalg.solve(G, U)                             # (100,3)

# ------------------------------ основная симуляция с PES ------------------------------
y   = np.zeros((N_t, 3))                              # (10001,3)
err = np.zeros((N_t, 3))                              # (10001,3)

V_pre = np.zeros(N_pre)                               # (100,)
ref    = np.zeros(N_pre)                              # (100,)
a      = np.zeros(N_pre)                              # (100,)
tau_psc = 0.25                                        # скаляр

spike_times = []                                      # список длины ~S (время спайков)
spike_ids   = []                                      # список длины ~S (индексы нейронов)

for n in range(1, N_t):
    x = u[n]                                          # (3,)

    # ток и мембранная динамика пред-ансамбля
    J = gain * (enc @ x) + bias                       # (100,) = (100,3)@(3,)
    active = (ref <= 0.0)                             # (100,) bool
    V_pre[active] += (dt / tau_rc) * (-V_pre[active] + J[active])  # (sum(active),)
    spikes = (V_pre >= 1.0)                           # (100,) bool

    if np.any(spikes):
        idx = np.flatnonzero(spikes)                  # (~S_n,)
        spike_times.append(np.full(len(idx), t[n]))   # (~S_n,)
        spike_ids.append(idx)                         # (~S_n,)

    # рефрактер и сброс
    ref[spikes] = tau_ref                             # (~S_n,)
    V_pre[spikes] = 0.0                               # (~S_n,)
    ref -= dt                                         # (100,)

    # экспоненциальная фильтрация импульсов → «мгновенная частота»
    a += (dt / tau_psc) * (-a + spikes / dt)          # (100,)

    # декодирование и постсинаптическая динамика узла y
    y_hat = a @ W                                     # (3,) = (100,)@(100,3)
    y[n]  = y[n-1] + (dt / tau_syn) * (y_hat - y[n-1])  # (3,)
    print(y.shape)
    # ошибка и обновление декодеров (PES)
    e =  x - y[n]                                     # (3,)
    err[n] = e                                        # (3,)
    W += (eta) * np.outer(a, e) * dt                 # (100,3) += (100,)*(3,) → (100,3)


# подготовка растр-данных
if spike_times:
    spike_times = np.concatenate(spike_times)         # (~S_total,)
    spike_ids   = np.concatenate(spike_ids)           # (~S_total,)
else:
    spike_times = np.array([])                        # (0,)
    spike_ids   = np.array([])                        # (0,)

fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# (a) фазовый портрет источника osc: u[:,0] vs u[:,1]
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(u[:, 0], u[:, 1], linewidth=2.0)
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.set_title("Фазовый портрет: osc\n (источник)")
ax1.axhline(0, linewidth=1); ax1.axvline(0, linewidth=1)
ax1.set_aspect("equal", "box")
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
ax1.legend(frameon=False)
ax1.grid(True, linestyle="--", alpha=0.35)

# (b) фазовый портрет копии osc2: y[:,0] vs y[:,1]
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(y[:, 0], y[:, 1], linewidth=2.0)
ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
ax2.set_title("Фазовый портрет: osc2\n (после обучения PES)")
ax2.axhline(0, linewidth=1); ax2.axvline(0, linewidth=1)
ax2.set_aspect("equal", "box")
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
ax2.legend(frameon=False)
ax2.grid(True, linestyle="--", alpha=0.35)

# (c) временные ряды ошибок PES
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, err[:, 0], linewidth=1.5, label="e₁")
ax3.plot(t, err[:, 1], linewidth=1.5, label="e₂")
ax3.plot(t, err[:, 2], linewidth=1.5, label="e₃")
ax3.set_xlabel("t, с"); ax3.set_ylabel("Ошибка e = x − y")
ax3.set_title("Компоненты ошибки PES")
ax3.legend(frameon=False)
ax3.grid(True, linestyle="--", alpha=0.35)

# (d) растровая диаграмма спайков пред-ансамбля
ax4 = fig.add_subplot(gs[1, 1])
if spike_times.size > 0:
    ax4.plot(spike_times, spike_ids, '|')
ax4.set_xlabel("Время, с"); ax4.set_ylabel("Индекс нейрона")
ax4.set_title(f"Спайковая активность")
ax4.grid(True, linestyle="--", alpha=0.35)

fig.suptitle("Обучение осциллятора с PES", y=0.98)
overview_path = "overview_pes.png"
plt.savefig(overview_path, dpi=200, bbox_inches="tight")
plt.show()
