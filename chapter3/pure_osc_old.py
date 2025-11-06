import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 30,
})

# ------------------------------ вспомогательные функции ------------------------------
def lif_rate(J, tau_rc=0.02, tau_ref=0.002):
    """
    Стационарная частота LIF (Гц) при постоянном входном токе J (порог = 1).
    """
    J = np.asarray(J)
    out = np.zeros_like(J, dtype=float)
    mask = J > 1.0
    if np.any(mask):
        JJ = J[mask]
        denom = tau_ref - tau_rc * np.log1p(-1.0 / JJ)
        out[mask] = 1.0 / denom
    return out

def current_from_rate(r, tau_rc=0.02, tau_ref=0.002):
    """
    Обратная функция: целевая частота r -> эквивалентный входной ток J.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-3)
    k = (1.0 / r - tau_ref) / tau_rc
    k = np.maximum(k, 1e-9)
    return 1.0 / (1.0 - np.exp(-k))

def unit_norm_rows(X, eps=1e-12):
    """
    L2-нормировка строк матрицы (нормируем энкодеры нейронов).
    """
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm

def sample_states_for_decoder(K, rng, radius_xy=1.0):
    """
    Обучающие состояния (x1,x2,s):
    (x1,x2) равномерно по диску радиуса radius_xy, s ~ U[0,1].
    """
    u = rng.random(K)
    r = radius_xy * np.sqrt(u)
    phi = rng.uniform(0, 2*np.pi, size=K)
    x1 = r * np.cos(phi)
    x2 = r * np.sin(phi)
    s  = rng.uniform(0.0, 1.0, size=K)
    X  = np.stack([x1, x2, s], axis=1)
    return X

def init_population(N, rng, tau_rc, tau_ref):
    """
    Создание параметров популяции LIF:
    энкодеры enc, усиления gain, смещения bias.
    """
    enc = unit_norm_rows(rng.normal(size=(N, 3)))       # (N,3)
    x_intercepts = rng.uniform(-1, 1, size=N)     # (N,)
    max_rates    = rng.uniform(100.0, 200.0, size=N)    # (N,)
    Jmax = current_from_rate(max_rates, tau_rc, tau_ref)# (N,)
    gain = (Jmax - 1.0) / (1.0 - x_intercepts)          # (N,)
    bias = 1.0 - gain * x_intercepts                    # (N,)
    return enc, gain, bias

def init_decoder(enc, gain, bias, X_samp, tau_rc, tau_ref):
    """
    Начальная оценка матрицы декодеров W (размерностью 3),
    решением МНК с Tikhonov-регуляризацией.
    """
    # Активность нейронов по обучающей выборке:
    A_samp = lif_rate(gain[:, None] * (enc @ X_samp.T) + bias[:, None],
                      tau_rc, tau_ref).T                # (K, N)
    G = A_samp.T @ A_samp + np.eye(A_samp.shape[1])  # (N,N)
    U = A_samp.T @ X_samp                                  # (N,3)
    W = np.linalg.solve(G, U)                              # (N,3)
    return W

# ------------------------------ параметры модели ------------------------------
rng = np.random.default_rng(1)

T, dt = 10.0, 0.001
t = np.arange(0.0, T + dt, dt)
N_t = t.size

omega   = 10.0
tau_syn = 0.1       # время фильтрации выходного контура
tau_rc  = 0.02      # мембранная постоянная
tau_ref = 0.002     # рефрактер (в динамике далее не используется)

# размеры трёх популяций
N1 = 100
N2 = 100
N3 = 100

eta = 1e-3          # скорость обучения PES на последнем слое

# ------------------------------ эталонная траектория u(t) ------------------------------
theta = omega * dt
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
u = np.zeros((N_t, 3))
u[0] = [1.0, 0.0, 1.0]
for n in range(1, N_t):
    u[n, :2] = R @ u[n-1, :2]
    u[n,  2] = 1.0

# ------------------------------ инициализация трех популяций ------------------------------
enc1, gain1, bias1 = init_population(N1, rng, tau_rc, tau_ref)
enc2, gain2, bias2 = init_population(N2, rng, tau_rc, tau_ref)
enc3, gain3, bias3 = init_population(N3, rng, tau_rc, tau_ref)

# обучающая выборка для МНК
K_samp = 4000
X_samp = sample_states_for_decoder(K_samp, rng, radius_xy=1.0)

W1 = init_decoder(enc1, gain1, bias1, X_samp, tau_rc, tau_ref)  # (N1,3)
W2 = init_decoder(enc2, gain2, bias2, X_samp, tau_rc, tau_ref)  # (N2,3)
W3 = init_decoder(enc3, gain3, bias3, X_samp, tau_rc, tau_ref)  # (N3,3)

# ------------------------------ динамические переменные ------------------------------
# мембранные потенциалы (без рефрактерных таймеров)
V1 = np.zeros(N1)
V2 = np.zeros(N2)
V3 = np.zeros(N3)

a1 = np.zeros(N1)
a2 = np.zeros(N2)
a3 = np.zeros(N3)

# выход и ошибка
y   = np.zeros((N_t, 3))
err = np.zeros((N_t, 3))

# ------------------------------ хранилища растра по популяциям ------------------------------
spike_times_1 = []
spike_ids_1   = []

spike_times_2 = []
spike_ids_2   = []

spike_times_3 = []
spike_ids_3   = []

# ------------------------------ основная симуляция ------------------------------
for n in range(1, N_t):
    x_true = u[n]  # целевой вектор (x1, x2, s)

    # ---- популяция 1: кодирует x_true ----
    J1 = gain1 * (enc1 @ x_true) + bias1          # (N1,)
    # без рефрактерного окна: все нейроны всегда интегрируют
    V1 += (dt / tau_rc) * (-V1 + J1)
    spikes1 = (V1 >= 1.0)

    # сохранить спайки pop1
    if np.any(spikes1):
        idx1 = np.flatnonzero(spikes1)
        spike_times_1.append(np.full(len(idx1), t[n]))
        spike_ids_1.append(idx1)

    # сброс потенциала после спайка (без "заморозки")
    V1[spikes1] = 0.0

    a1 += dt * (-a1 + spikes1 / dt)   # (N1,)
    x_hat1 = a1 @ W1                  # (3,)

    # ---- популяция 2: получает x_hat1 ----
    J2 = gain2 * (enc2 @ x_hat1) + bias2          # (N2,)
    V2 += (dt / tau_rc) * (-V2 + J2)
    spikes2 = (V2 >= 1.0)

    if np.any(spikes2):
        idx2 = np.flatnonzero(spikes2)
        spike_times_2.append(np.full(len(idx2), t[n]))
        spike_ids_2.append(idx2)

    V2[spikes2] = 0.0

    a2 += dt * (-a2 + spikes2 / dt)   # (N2,)
    x_hat2 = a2 @ W2                  # (3,)

    # ---- популяция 3: получает x_hat2 ----
    J3 = gain3 * (enc3 @ x_hat2) + bias3          # (N3,)
    V3 += (dt / tau_rc) * (-V3 + J3)
    spikes3 = (V3 >= 1.0)

    if np.any(spikes3):
        idx3 = np.flatnonzero(spikes3)
        spike_times_3.append(np.full(len(idx3), t[n]))
        spike_ids_3.append(idx3)

    V3[spikes3] = 0.0

    a3 += dt * (-a3 + spikes3 / dt)   # (N3,)
    y_hat = a3 @ W3                   # (3,)

    # выходная динамика (медленная фильтрация)
    y[n] = y[n-1] + (dt / tau_syn) * (y_hat - y[n-1])

    # ошибка и PES-обновление декодера последней популяции
    e = x_true - y[n]   # (3,)
    err[n] = e
    W3 += eta * np.outer(a3, e) * dt  # обучение по правилу PES

# подготовка растр-данных для визуализации
def concat_spikes(times_list, ids_list):
    if times_list:
        return np.concatenate(times_list), np.concatenate(ids_list)
    else:
        return np.array([]), np.array([])

spk_t1, spk_i1 = concat_spikes(spike_times_1, spike_ids_1)
spk_t2, spk_i2 = concat_spikes(spike_times_2, spike_ids_2)
spk_t3, spk_i3 = concat_spikes(spike_times_3, spike_ids_3)

# ------------------------------ визуализация ------------------------------
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# (a) фазовый портрет источника
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(u[:, 0], u[:, 1], linewidth=2.0)
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.set_title("Фазовый портрет:\n osc1")
ax1.axhline(0, linewidth=1); ax1.axvline(0, linewidth=1)
ax1.set_aspect("equal", "box")
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle="--", alpha=0.35)

# (b) фазовый портрет результата после pop3
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(y[:, 0], y[:, 1], linewidth=2.0)
ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
ax2.set_title("Фазовый портрет:\n osc2")
ax2.axhline(0, linewidth=1); ax2.axvline(0, linewidth=1)
ax2.set_aspect("equal", "box")
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle="--", alpha=0.35)

# (c) компоненты ошибки e = x_true - y
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(t, err[:, 0], linewidth=1.5, label="e₁")
ax3.plot(t, err[:, 1], linewidth=1.5, label="e₂")
ax3.plot(t, err[:, 2], linewidth=1.5, label="e₃")
ax3.set_xlabel("t, с"); ax3.set_ylabel("Ошибка e")
ax3.set_title("Компоненты ошибки PES")
ax3.legend(frameon=False)
ax3.grid(True, linestyle="--", alpha=0.35)

# (d) растр популяции 1
ax4 = fig.add_subplot(gs[1, 0])
if spk_t1.size > 0:
    ax4.plot(spk_t1, spk_i1, '|')
ax4.set_xlabel("Время, с"); ax4.set_ylabel("Нейроны osc1")
ax4.set_title("Спайковая активность\n o1")
ax4.set_ylim(-5, N1 + 5)
ax4.grid(True, linestyle="--", alpha=0.35)

# (e) растр популяции 2
ax5 = fig.add_subplot(gs[1, 1])
if spk_t2.size > 0:
    ax5.plot(spk_t2, spk_i2, '|')
ax5.set_xlabel("Время, с"); ax5.set_ylabel("Нейроны osc2")
ax5.set_title("Спайковая активность\n o2")
ax5.set_ylim(-5, N2 + 5)
ax5.grid(True, linestyle="--", alpha=0.35)

# (f) растр популяции 3
ax6 = fig.add_subplot(gs[1, 2])
if spk_t3.size > 0:
    ax6.plot(spk_t3, spk_i3, '|')
ax6.set_xlabel("Время, с"); ax6.set_ylabel("Нейроны err")
ax6.set_title("Спайковая активность\n e")
ax6.set_ylim(-5, N3 + 5)
ax6.grid(True, linestyle="--", alpha=0.35)

fig.suptitle("Обучение осциллятора с PES", y=0.99)
plt.savefig("overview_pes.png", dpi=200, bbox_inches="tight")
plt.show()
