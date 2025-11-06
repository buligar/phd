import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 30,
})

# Программа предназначена для моделирования популяций импульсных LIF-нейронов и
# онлайн-обучения линейного декодера состояния осциллятора по правилу Prescribed Error Sensitivity.
# Программа генерирует опорную траекторию гармонического осциллятора, кодирует её в виде
# спайковой активности, восстанавливает состояние и корректирует веса декодера пропорционально
# вектору ошибки воспроизведения. Программа может использоваться в прикладных исследованиях
# нейроморфных алгоритмов управления, копирования динамики и слежения за сигналом, а также
# для демонстрации принципов обучения с ошибкой в терминах нейронного инженерного
# фреймворка. Функциональные возможности программы включают: реконструкцию сигнала из
# спайков; адаптацию весов декодера по сигналу ошибки; построение фазовых портретов
# эталонного и восстановленного осцилляторов; анализ временной эволюции ошибки; визуализацию спайковой активности нейронов.


def current_from_rate(r, tau_rc=0.02, tau_ref=0.002):
    k = (1.0 / r - tau_ref) / tau_rc # (100,)
    return 1.0 / (1.0 - np.exp(-k)) # (100,)


def init_population(N, rng, tau_rc, tau_ref):
    enc = rng.normal(size=(N, 2)) # (100, 2) 
    x_intercepts = rng.uniform(-1, 1, size=N) # (100,) 
    max_rates    = rng.uniform(100.0, 200.0, size=N) # (100,)    
    Jmax = current_from_rate(max_rates, tau_rc, tau_ref) # (100,)
    gain = (Jmax - 1.0) / (1.0 - x_intercepts) # (100,)
    bias = 1.0 - gain * x_intercepts # (100,)            
    return enc, gain, bias # (100, 2), (100,), (100,)

rng = np.random.default_rng(1)

T, dt = 1.0, 0.001
t = np.arange(0.0, T + dt, dt) # (10001,)
N_t = t.size # 10001

omega   = 10.0
tau_syn = 0.05  
tau_rc  = 0.02      # постоянная времени мембраны
tau_ref = 0.002     # рефрактерный период

# размеры популяций
N1 = 200
N2 = 200
N3 = 200

eta = 1e-3 # скорость обучения PES

theta = omega * dt
M = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]]) # матрица поворота
u = np.zeros((N_t, 2)) # массив нулей (10001, 2)
u[0] = [1, 0]
for n in range(1, N_t):
    u[n] = M @ u[n-1] # изменяем первые две координаты x1, x2


enc1, gain1, bias1 = init_population(N1, rng, tau_rc, tau_ref) # (100, 2) (100,) (100,)
enc2, gain2, bias2 = init_population(N2, rng, tau_rc, tau_ref) # (100, 2) (100,) (100,)
enc3, gain3, bias3 = init_population(N3, rng, tau_rc, tau_ref) # (100, 2) (100,) (100,)


W1 = np.zeros((N1, 2)) # массив нулей (100, 2)
W2 = np.zeros((N2, 2)) # массив нулей (100, 2) 
W3 = np.zeros((N3, 2)) # массив нулей (100, 2)

V1 = np.zeros(N1) # массив нулей (100,)
V2 = np.zeros(N2) # массив нулей (100,)
V3 = np.zeros(N3) # массив нулей (100,)

a1 = np.zeros(N1) # массив нулей (100,)
a2 = np.zeros(N2) # массив нулей (100,)
a3 = np.zeros(N3) # массив нулей (100,)

y   = np.zeros((N_t, 2)) # массив нулей (10001, 2)
err = np.zeros((N_t, 2)) # массив нулей (10001, 2)

spike_times_1 = []
spike_ids_1   = []

spike_times_2 = []
spike_ids_2   = []

spike_times_3 = []
spike_ids_3   = []

for n in range(1, N_t):
    J1 = gain1 * (enc1 @ u[n]) + bias1 # (100,)
    V1 += (dt / tau_rc) * (-V1 + J1) # (100,)
    spikes1 = (V1 >= 1.0) # (100,)

    if np.any(spikes1):
        idx1 = np.flatnonzero(spikes1) 
        spike_times_1.append(np.full(len(idx1), t[n])) # переменные массивы времени спайков
        spike_ids_1.append(idx1) # переменные массивы id спайков

    V1[spikes1] = 0.0
     
    J2 = gain2 * (enc2 @ y[n-1]) + bias2 # (100,)        
    V2 += (dt / tau_rc) * (-V2 + J2) # (100,)
    spikes2 = (V2 >= 1.0) # (100,)

    if np.any(spikes2):
        idx2 = np.flatnonzero(spikes2)
        spike_times_2.append(np.full(len(idx2), t[n])) # переменные массивы времени спайков
        spike_ids_2.append(idx2) # переменные массивы id спайков

    V2[spikes2] = 0.0
     
    a2 += dt * (-a2 / tau_syn) + spikes2.astype(float) / tau_syn
    y_hat = a2 @ W2

    y[n] = y[n-1] + (dt / tau_syn) * (y_hat - y[n-1])

    e = u[n] - y[n]
    err[n] = e # (10001, 2)
    W2 += eta * np.outer(a2, e) * dt  # обучение по правилу PES # (100,2)


    J3 = gain3 * (enc3 @ y[n]) + bias3 # (100,)         
    V3 += (dt / tau_rc) * (-V3 + J3) # (100,)
    spikes3 = (V3 >= 1.0) # (100,)

    if np.any(spikes3):
        idx3 = np.flatnonzero(spikes3)
        spike_times_3.append(np.full(len(idx3), t[n])) # переменные массивы времени спайков
        spike_ids_3.append(idx3) # переменные массивы id спайков

    V3[spikes3] = 0.0
           

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
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_title("Фазовый портрет:\n $o_1$")
ax1.axhline(0, linewidth=1); ax1.axvline(0, linewidth=1)
ax1.set_aspect("equal", "box")
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle="--", alpha=0.35)

# (b) компоненты ошибки e = u - y
ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(t, err[:, 0], linewidth=1.5, label="x")
ax3.plot(t, err[:, 1], linewidth=1.5, label="y")
ax3.set_xlabel("t, с"); ax3.set_ylabel("Ошибка $e$")
ax3.set_title("Компоненты ошибки PES")
ax3.legend(frameon=False)
ax3.grid(True, linestyle="--", alpha=0.35)

# (c) фазовый портрет результата
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(y[:, 0], y[:, 1], linewidth=2.0)
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_title("Фазовый портрет:\n $o_2$")
ax2.axhline(0, linewidth=1); ax2.axvline(0, linewidth=1)
ax2.set_aspect("equal", "box")
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle="--", alpha=0.35)


# (d) растр популяции o_1
ax4 = fig.add_subplot(gs[1, 0])
if spk_t1.size > 0:
    ax4.plot(spk_t1, spk_i1, '|')
ax4.set_xlabel("Время, с"); ax4.set_ylabel("Нейроны $o_1$")
ax4.set_title("Спайковая активность\n $o_1$")
ax4.set_ylim(-5, N1 + 5)
ax4.grid(True, linestyle="--", alpha=0.35)

# (e) растр популяции o_2
ax5 = fig.add_subplot(gs[1, 1])
if spk_t2.size > 0:
    ax5.plot(spk_t2, spk_i2, '|')
ax5.set_xlabel("Время, с"); ax5.set_ylabel("Нейроны $e$")
ax5.set_title("Спайковая активность\n $e$")
ax5.set_ylim(-5, N2 + 5)
ax5.grid(True, linestyle="--", alpha=0.35)

# (f) растр популяции e
ax6 = fig.add_subplot(gs[1, 2])
if spk_t3.size > 0:
    ax6.plot(spk_t3, spk_i3, '|')
ax6.set_xlabel("Время, с"); ax6.set_ylabel("Нейроны $o_2$")
ax6.set_title("Спайковая активность\n $o_2$")
ax6.set_ylim(-5, N3 + 5)
ax6.grid(True, linestyle="--", alpha=0.35)

plt.savefig("overview_pes.png", dpi=200, bbox_inches="tight")
plt.show()

