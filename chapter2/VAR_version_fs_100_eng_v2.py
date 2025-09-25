from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg as npl

start_scope()

# Параметры моделирования
sampling_frequency = 100  # Гц (100 - чтобы увидеть частоты 0-40, 200 чтобы увидеть частоты 0-80)
dt_sim = 1 / sampling_frequency  # шаг интегрирования (сек)
defaultclock.dt = dt_sim * second

# Временные параметры
t_total_sim = 10   # общее время моделирования (сек)
segment_switch = 5  # время переключения драйвера (сек)

n_neurons = 100
fs = sampling_frequency
n_samples = int(fs * t_total_sim)
A = 100 * pA
R = 80 * Mohm
f = 10*Hz  
f2 = 30*Hz     
tau = 20*ms
phi = 0
J = 1 * mV
eqs = '''
dv/dt = (v_rest-v+R*(I_half1+I_half2))/tau :  volt            
I_half1 = int(t < 5000*ms) * amplitude * sin(2*pi*f*t + phi) : amp              
I_half2 = int(t >= 5000*ms) * amplitude2 * sin(2*pi*f2*t + phi) : amp
amplitude : amp 
amplitude2 : amp 
'''
v_threshold = -50 * mV
v_reset = -70 * mV
v_rest =  -65 * mV

G = NeuronGroup(n_neurons, 
                eqs, 
                threshold="v > v_threshold",
                reset="v = v_reset",
                method='euler')
G.v = v_rest

G.amplitude = A  
G.amplitude2 = A


# Вероятности связей
p_intra1 = 0.15
p_intra2 = 0.15
p_12     = 0.05
p_21     = 0.05

# Веса соединения
w_intra1 = 7
w_intra2 = 7
w_12     = 7
w_21     = 7

n_half = n_neurons // 2

n_steps      = int(t_total_sim * fs)  # число шагов интегрирования

# 2) Задаём два массива скоростей: rate1 для нейронов 0–49, rate2 для 50–99
rate1_array = np.zeros(n_steps)*Hz
rate2_array = np.zeros(n_steps)*Hz

#   - в интервале t=[0,5) сек: rate1=10 Гц, rate2=60 Гц
rate1_array[:int(5*fs)] = 10*Hz
rate2_array[:int(5*fs)] = 60*Hz
#   - в интервале t=[5,10) сек: rate1=60 Гц, rate2=10 Гц
rate1_array[int(5*fs):] = 60*Hz
rate2_array[int(5*fs):] = 10*Hz

rate1_t = TimedArray(rate1_array, dt=dt_sim*second)
rate2_t = TimedArray(rate2_array, dt=dt_sim*second)

n_half = n_neurons // 2
P1 = PoissonGroup(n_half, rates='rate1_t(t)')
P2 = PoissonGroup(n_half, rates='rate2_t(t)')

# P1 → G[0:50],  P2 → G[50:100]
syn1 = Synapses(P1, G[:n_half], on_pre='v_post += J')
syn1.connect(p=0.3)
syn2 = Synapses(P2, G[n_half:], on_pre='v_post += J')
syn2.connect(p=0.3)

sizes = [50, 50]
probs = [[0.15, 0.05], [0.05, 0.15]]
g = nx.stochastic_block_model(sizes, probs, directed=False)

# ===================== SBM → E/I cинапсы (замена блоков 5.1–5.8) =====================
# Индексы E/I как в вашей постановке: 0–39 и 50–89 — E; 40–49 и 90–99 — I
exc_idx = np.r_[0:40, 50:90]
inh_idx = np.r_[40:50, 90:100]

# Метки блоков для выбора веса: 0 для [0..49], 1 для [50..99]
block_id = np.zeros(n_neurons, dtype=int)
block_id[50:] = 1

# Забираем рёбра ненаправленного графа и создаём пары в обоих направлениях
edges = np.array(g.edges(), dtype=int)  # shape: (m, 2), без петель
pre  = np.r_[edges[:, 0], edges[:, 1]]
post = np.r_[edges[:, 1], edges[:, 0]]

# Назначение весов по направлению и принадлежности к блокам
bpre  = block_id[pre]
bpost = block_id[post]
# Внутри блока 1 -> w_intra1; внутри блока 2 -> w_intra2; 0->1 -> w_12; 1->0 -> w_21
w_all = np.where(
    (bpre == 0) & (bpost == 0), w_intra1,
    np.where(
        (bpre == 1) & (bpost == 1), w_intra2,
        np.where((bpre == 0) & (bpost == 1), w_12, w_21)
    )
).astype(float)

is_exc = np.isin(pre, exc_idx)

pre_exc,  post_exc,  w_exc  = pre[is_exc],  post[is_exc],  w_all[is_exc]
pre_inh,  post_inh,  w_inh  = pre[~is_exc], post[~is_exc], w_all[~is_exc]

# Создаём две группы синапсов: E→* и I→*
S_E = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_E.connect(i=pre_exc, j=post_exc)
S_E.w = w_exc

S_I = Synapses(G, G, model='w : 1', on_pre='v_post -= J * w')
S_I.connect(i=pre_inh, j=post_inh)
S_I.w = w_inh

W = np.zeros((n_neurons, n_neurons))
W[S_E.i[:], S_E.j[:]] = S_E.w[:]
W[S_I.i[:], S_I.j[:]] = S_I.w[:]


fig, ax = plt.subplots(figsize=(6, 5))

im = ax.matshow(W, cmap='viridis')          # сам образ (AxesImage)
cbar = fig.colorbar(im, ax=ax,             # ← добавляем colorbar
                    shrink=0.85,           # (необязательно) делаем короче
                    pad=0.02)              # (необязательно) отступ от осей
cbar.set_label('Weight of synapses', fontsize=16)

ax.set_xlabel('Postsynaptic neuron', fontsize=16)
ax.set_ylabel('Presynaptic neuron', fontsize=16)
ax.tick_params(axis='x', which='major', labelsize=12)
ax.tick_params(axis='y', which='major', labelsize=12)


fig.savefig('spike_raster_2col.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight')
fig.savefig('spike_raster_2col.png',
            format='png',
            dpi=600,
            bbox_inches='tight')

mon = StateMonitor(G, 'v', record=True)
spike_monitor = SpikeMonitor(G)

# Запуск моделирования
run(t_total_sim * second)

spike_times = spike_monitor.t / second
spike_indices = spike_monitor.i

# --- Построение спайкового растера в двухколоночном формате IEEE ---
plt.figure(figsize=(10, 8))
plt.scatter(spike_times, spike_indices, marker='|')
plt.xlim(0,t_total_sim)
plt.ylim(0,n_neurons)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Neurons', fontsize=16)
plt.tick_params(axis='x', which='major', labelsize=14)
plt.tick_params(axis='y', which='major', labelsize=14)
plt.savefig('spikes.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('spikes.png', format='png', dpi=600, bbox_inches='tight')
plt.show()



# Извлечение данных
t_sim = mon.t/second
x1 = mon.v[:n_neurons//2, :] / mV  # (форма: n_neurons//2, 1000)
x2 = mon.v[n_neurons//2:, :] / mV  # (форма: n_neurons//2, 1000)

trial0 = x1.T  # (1000, n_neurons//2)
trial1 = x2.T  # (1000, n_neurons//2)
signal_noisy = np.stack((trial0, trial1), axis=-1)  # (1000, n_neurons//2, 2)

v1 = mon.v[:n_neurons//2 , :].mean(axis=0) / mV             # кластер 1
v2 = mon.v[n_neurons//2:, :].mean(axis=0) / mV              # кластер 2


trial0 = x1.T  # (1000, n_neurons//2)
trial1 = x2.T  # (1000, n_neurons//2)
signal_noisy = np.stack((trial0, trial1), axis=-1)  # (1000, n_neurons//2, 2)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_sim, signal_noisy[:,0,0], label="Сигнал 1")
axes[0].set_xlabel("Time (s)", fontsize=16)
axes[0].set_ylabel("Membrane potential (mV)", fontsize=16)
axes[0].legend()

axes[1].plot(t_sim, signal_noisy[:,0,1], label="Сигнал 2")
axes[1].set_xlabel("Time (s)", fontsize=16)
axes[1].set_ylabel("Membrane potential (mV)", fontsize=16)
axes[1].legend()



v  = np.stack([v1, v2], axis=-1)



def fit_var_ols_const(y, p, ridge=0.0):
    y = np.asarray(y, float)
    T, k = y.shape; N = T - p
    if N <= (k*p + 1): raise ValueError("Слишком короткое окно для заданного p.")
    Y = y[p:]; lags = [y[p-l-1:T-l-1] for l in range(p)]
    X = np.hstack([np.ones((N,1)), *lags])
    XtX, XtY = X.T @ X, X.T @ Y
    B = npl.solve(XtX + ridge*np.eye(XtX.shape[0]), XtY) if ridge>0 else npl.solve(XtX, XtY)
    c = B[0, :]
    A = np.stack([B[1+i*k:1+(i+1)*k, :].T for i in range(p)], axis=0)
    E = Y - X @ B
    Sigma = (E.T @ E) / N
    return A, c, Sigma

def companion_matrix(A):
    p, k, _ = A.shape
    top = np.hstack(A)
    if p == 1: 
        return top
    I = np.eye(k*(p-1))
    Z = np.zeros((k*(p-1), k))
    return np.vstack([top, np.hstack([I, Z])])

def is_stable(A, tol=1.0-1e-10):
    Φ = companion_matrix(A)
    return np.max(np.abs(npl.eigvals(Φ))) < tol

def information_criteria(y, p_max, ridge=0.0):
    y = np.asarray(y, float); T, k = y.shape
    AIC = []; BIC = []; HQ = []
    for p in range(1, p_max+1):
        try:
            A, c, Sigma = fit_var_ols_const(y, p, ridge=ridge)
            if not is_stable(A): raise RuntimeError
            N = T - p; 
            q = k*k*p + k
            ld = np.log(npl.det(Sigma)).real
            AIC.append(ld + (2.0*q)/N)
            BIC.append(ld + (np.log(N)*q)/N)
            HQ .append(ld + (2.0*np.log(np.log(N))*q)/N)
        except Exception:
            AIC.append(np.inf); BIC.append(np.inf); HQ.append(np.inf)
    return np.array(AIC), np.array(BIC), np.array(HQ)

def choose_var_order(v, win_len, step, p_max=20, criterion='BIC', ridge=0.0, aggregator='mode'):
    pstars = []
    for start in range(0, v.shape[0] - win_len + 1, step):
        seg = v[start:start + win_len]
        AIC, BIC, HQ = information_criteria(seg, p_max, ridge=ridge)
        ic = {'AIC': AIC, 'BIC': BIC, 'HQ': HQ}[criterion.upper()]
        if np.isfinite(ic).any():
            pstars.append(1 + np.argmin(ic))
    pstars = np.array(pstars, int)
    if pstars.size == 0:
        raise RuntimeError("Нет валидных окон для выбора порядка.")
    if aggregator == 'median':
        p_global = int(np.round(np.median(pstars)))
    elif aggregator == 'mean':
        p_global = int(np.round(np.mean(pstars)))
    else:  # mode
        cnt = np.bincount(pstars, minlength=p_max+1); cnt[0] = 0
        p_global = np.flatnonzero(cnt == cnt.max()).min()
    print(f"Рекомендуемый порядок VAR по {criterion}: p = {p_global}")
    print(f"Медиана по окнам: {int(np.median(pstars))}, среднее: {np.mean(pstars):.2f}")
    return p_global, pstars

p_opt, p_all = choose_var_order(v, win_len=50, step=16, p_max=20, criterion='BIC', ridge=0.0, aggregator='mode')

win_len, step, order = 50, 16, p_opt # p=3
nfft    = 128                   # более тонкая сетка
freqs   = np.linspace(0, fs/2, nfft//2+1)
n_freq    = len(freqs) # 51

def spectra_2x2(A, Sigma, i, j):
    nF = len(freqs)
    gc  = np.empty(nF)
    dtf = np.empty(nF)
    pdc = np.empty(nF)

    # Компоненты ковариационной матрицы
    s11, s22 = Sigma[0, 0], Sigma[1, 1]
    s12, s21 = Sigma[0, 1], Sigma[1, 0]

    # Ортогонализация для GC
    if (i, j) == (0, 1): # 2->1 
        T = np.array([[1.0, -s12/s22],
                      [0.0,  1.0]])
    elif (i, j) == (1, 0): # 1->2 
        T = np.array([[1.0,  0.0], 
                      [-s21/s11, 1.0]])
    else:
        raise ValueError("Индексы i, j должны быть 0/1 и i != j.")

    print(len(freqs))
    for l, f in enumerate(freqs): # l=51 f=0-50
        z  = np.exp(-1j * 2*np.pi * f / fs)
        Az = np.eye(2, dtype=complex) # [[1+j, 0+j],[0+j, 1+j]]
        for p, Al in enumerate(A, start=1): # p=4 Al=(2,2)
            Az -= Al * z**p

        Hz = np.linalg.inv(Az) # Az=[[a,b][c,d]] A(z)^{-1}=1/(ad-bc)[[d, -b],[-c, a]]
        Sz = Hz @ Sigma @ Hz.conj().T # S(z)=H(z)ΣH^*(z) Hz=[[3+j, 5],[2-j, j]] Hz^*=[[3-j, 2+j],[5, -j]]
        H_t      = T @ Hz 
        Sigma_t  = T @ Sigma @ T.T
        Sii_cond = np.real(Sigma_t[i, i]) * (np.abs(H_t[i, i])**2)
        gc[l] = np.log(np.real(Sz[i, i]) / Sii_cond) # ln(S_11(w)/S_(11|2)(w)) np.real - берем действительную часть числа (без j)
        dtf[l] = np.abs(Hz[i, j])**2 / (np.abs(Hz[i, 0])**2 + np.abs(Hz[i, 1])**2)  # |Hz_ij|^2/(|Hz_i0|^2+|Hz_i1|^2)
        pdc[l] = np.abs(Az[i, j])**2 / (np.abs(Az[0, j])**2 + np.abs(Az[1, j])**2)  # |Az_ij|^2/(|Az_0j|^2+|Az_1j|^2)


    return gc, dtf, pdc


def var_ols_const(y, p=4):
    y = np.asarray(y, float)
    # y = y.reshape(50, 100)
    T, k = y.shape # (50, 2)
    Y = y[p:] # (47, 2)
    lags = [y[p-l-1:T-l-1] for l in range(p)] # (3, 47, 2) -> смещенные сигналы
    X = np.hstack([np.ones((T-p, 1)), *lags]) # (47, 7) -> N=T-p=47, 1+k*p=1+2*3=7 -> массив единиц

    # β = (XᵀX)⁻¹XᵀY
    XtX = X.T @ X # (7, 7)
    XtY = X.T @ Y # (7, 2)
    alpha = 1e-3  # или подобрать на кросс‐валидации
    B = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), XtY) # (7, 2) решаем Ax=b, где XtX-A, XtY-b и еще добавляем регуляризацию
    # B = np.linalg.solve(XtX, XtY) # (7, 2) решаем Ax=b, где XtX-A, XtY-b
    B_new = [B[1+i*k : 1+(i+1)*k].T for i in range(p)] # (3, 2, 2)
    A = np.stack(B_new, axis=0) # (3, 2, 2) -> собираем в один тензор
    
    E = Y - X @ B # (47, 2)
    dof = (T-p) - (k*p + 1) # 40
    Sigma = (E.T @ E) / dof # (2, 2)
    return A, Sigma # (4, 2, 2) и (2, 2)




def slide_2x2(i, j):
    G, D, P, T = [], [], [], []
    for start in range(0, n_samples - win_len + 1, step): # 50 (0-745)
        seg = v[start:start + win_len]
        A, Sigma = var_ols_const(seg, order)
        g, d, p = spectra_2x2(A, Sigma, i, j)
        G.append(g) 
        D.append(d) 
        P.append(p)
        T.append((start + win_len//2) / fs)
    return np.array(G).T, np.array(D).T, np.array(P).T, np.array(T)


gc_12, dtf_12, pdc_12, times = slide_2x2(1, 0)
gc_21, dtf_21, pdc_21, _ = slide_2x2(0, 1)


n_win  = gc_12.shape[1]              
dt      = step / fs                   
times   = np.arange(n_win)*dt + win_len/(2*fs)     
time_edges = np.arange(n_win + 1)*dt            
df         = freqs[1] - freqs[0]
freq_edges = np.concatenate(([0], freqs[:-1] + df/2, [freqs[-1] + df/2]))

measures = {
    'GC' : (gc_12,  gc_21),
    'PDC': (pdc_12, pdc_21),
    'DTF': (dtf_12, dtf_21)
}

fig, axs = plt.subplots(3, 2,
                        figsize=(7, 12),
                        sharex=True, sharey=True,
                        constrained_layout=True)

measures = {
    'GC' : (gc_12,  gc_21),
    'PDC': (pdc_12, pdc_21),
    'DTF': (dtf_12, dtf_21)
}

for row, (name, (m12, m21)) in enumerate(measures.items()):
    im1 = axs[row, 0].pcolormesh(time_edges, freq_edges, m12,
                                 shading='auto', cmap='turbo')
    axs[row, 0].set_title(f'{name}, 1 → 2', fontsize=14)
    im2 = axs[row, 1].pcolormesh(time_edges, freq_edges, m21,
                                 shading='auto', cmap='turbo')
    axs[row, 1].set_title(f'{name}, 2 → 1', fontsize=14)
    cbar = fig.colorbar(im1, ax=[axs[row,0], axs[row,1]],
                        shrink=0.85, pad=0.02)
    cbar.set_label('Power', fontsize=12)

for ax in axs[:, 0]:
    ax.set_xlabel('Time (s)', fontsize=12)
for ax in axs[:, 1]:
    ax.set_xlabel('Time (s)', fontsize=12)
for ax in axs[:, 0]:
    ax.set_ylabel('Frequency (Hz)', fontsize=12)

# общие границы и шаг тиков
for ax in axs.flatten():
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2.0))   # тики каждые 2 с
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1.0))   # минорные тики

for row in (0, 1):
    for col in (0, 1):
        ax = axs[row, col]
        ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)

fig.savefig('figure_2col.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig('figure_2col.png', format='png', dpi=600, bbox_inches='tight')



