import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import pickle

# --- Параметры моделирования и пути к результатам ---
results_dirs = {
    'proxy centrality\nбез буста':    'results_ext_test1_random_neighbors_bez_boost/rates',
    'proxy centrality\nс бустом':      'results_ext_test1_random_neighbors_s_boost/rates',
    'top centrality\nбез буста':      'results_ext_test1_one_cluster_bez_boost/rates',
    'top centrality\nс бустом':        'results_ext_test1_one_cluster_s_boost/rates',
    'hybrid centrality\nбез буста':    'results_ext_test1_top_neighbors_bez_boost/rates',
    'hybrid centrality\nс бустом':     'results_ext_test1_top_neighbors_s_boost/rates',
}

p_within  = 0.15  # вероятность связи внутри кластеров
p_between = 0.10  # вероятность связи между кластерами
p_input   = 0.20  # вероятность входа от внешнего стимула
f_max     = 50.0  # Гц, предел по оси частот
f0        = 10.0  # Гц, центральная частота сигнала
delta_f   = 1.0   # Гц, ширина полосы ±1 Гц

measures = [
    'betweenness',
    'percolation',
    'closeness',
    'harmonic',
    'degree',
    'eigenvector',
    'random',
]

# --- Функции для загрузки и анализа ---
def load_group2(fname):
    if isinstance(fname, Path):
        fname = fname.as_posix()
    opener = gzip.open if fname.endswith('.gz') else open
    with opener(fname, 'rb') as f:
        data = pickle.load(f)
    return data['t_group2'], data['rate_group2']

def calc_psd(t_ms, rate_hz, f_lim=f_max):
    rate = rate_hz - np.mean(rate_hz)
    N = len(rate)
    if N < 2:
        return np.array([]), np.array([])
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt
    window = np.hanning(N)
    S = np.abs(np.fft.rfft(rate * window)) / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mask = freqs <= f_lim
    return freqs[mask], S[mask]

# --- Сбор SNR по всем мерам и конфигурациям ---
snr_results = []
for measure in measures:
    freqs = None
    for label, dir_path in results_dirs.items():
        pattern = (
            f"rates_within{p_within:.2f}_between{p_between:.2f}_"
            f"input{p_input:.2f}_{measure}_test*.pickle"
        )
        files = sorted(Path(dir_path).glob(pattern))
        if not files:
            continue
        all_S = []
        for fn in files:
            t, r = load_group2(fn)
            f, S = calc_psd(t, r)
            all_S.append(S)
        if not all_S:
            continue
        arr = np.vstack(all_S)
        mean_S = np.mean(arr, axis=0)
        if freqs is None:
            freqs = f
        # signal vs noise
        sig_idx   = (freqs >= f0 - delta_f) & (freqs <= f0 + delta_f)
        noise_idx = ~sig_idx
        P_signal = np.trapz(mean_S[sig_idx], freqs[sig_idx])
        P_noise  = np.mean(mean_S[noise_idx])
        SNR_dB = 10 * np.log10(P_signal / P_noise)
        snr_results.append({
            'measure': measure,
            'configuration': label,
            'snr_db': round(SNR_dB, 2)
        })

# Преобразуем в словарь для быстрого доступа
snr_dict = {
    (row['measure'], row['configuration']): row['snr_db']
    for row in snr_results
}

# --- Построение 3D-графиков PSD с аннотациями SNR в 2D ---
fig, axs = plt.subplots(
    nrows=4, ncols=2,
    subplot_kw={'projection': '3d'},
    figsize=(12, 18),
)
axs = axs.flatten()

for ax, measure in zip(axs, measures):
    freqs = None
    stats = []

    # Считываем PSD
    for label, dir_path in results_dirs.items():
        pattern = (
            f"rates_within{p_within:.2f}_between{p_between:.2f}_"
            f"input{p_input:.2f}_{measure}_test*.pickle"
        )
        files = sorted(Path(dir_path).glob(pattern))
        if not files:
            continue
        all_S = []
        for fn in files:
            t, r = load_group2(fn)
            f, S = calc_psd(t, r)
            all_S.append(S)
        if not all_S:
            continue
        arr = np.vstack(all_S)
        mean_S = np.mean(arr, axis=0)
        std_S  = np.std(arr, axis=0)
        stats.append((label, mean_S, std_S))
        if freqs is None:
            freqs = f

    # Линии и поверхности ±σ
    for idx, (label, mean_S, std_S) in enumerate(stats):
        y_pos = idx * 2
        ax.plot(freqs, np.full_like(freqs, y_pos), mean_S,
                linewidth=2, color=f'C{idx}')
        F = np.vstack([freqs, freqs])
        Y = np.full_like(F, y_pos)
        Z = np.vstack([mean_S - std_S, mean_S + std_S])
        ax.plot_surface(F, Y, Z, alpha=0.3, color=f'C{idx}', shade=False)

    # 2D-аннотация SNR (справа вверху)
    snr_lines = []
    for idx, (label, _, _) in enumerate(stats):
        val = snr_dict.get((measure, label))
        if val is not None:
            snr_lines.append(f"{idx+1}: {val:.2f} dB")
    text2d = "\n".join(snr_lines)
    ax.text2D(
        0.02, 0.90, text2d,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

    # Оформление
    ax.set_title(measure, fontsize=16, y=0.9)
    ax.set_xlim(0, f_max)
    ax.set_zlim(0, 5)
    ax.set_xlabel('Частота, Гц', fontsize=14)
    ax.set_ylabel('Конфиг. (индекс)', fontsize=14)
    ax.set_zlabel('PSD', fontsize=14)
    ax.set_yticks(np.arange(len(results_dirs)) * 2)
    ax.set_yticklabels([str(i+1) for i in range(len(results_dirs))],
                       fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    ax.view_init(elev=10, azim=-30)

# Легенда в пустой оси
ax_leg = axs[len(measures)]
ax_leg.axis('off')
from matplotlib.lines import Line2D
keys_clean = [k.replace('\n',' ') for k in results_dirs.keys()]
handles = [Line2D([0],[0], color=f'C{i}', lw=2)
           for i in range(len(keys_clean))]
labels = [f"{i+1} — {keys_clean[i]}" for i in range(len(keys_clean))]
ax_leg.legend(handles, labels,
              title='Конфигурации',
              loc='center',
              fontsize=12,
              title_fontsize=14)

# Общий заголовок и сохранение
fig.subplots_adjust(left=0, right=0.88, bottom=0, top=0.98,
                    wspace=0, hspace=0)
fig.suptitle(
    f'PSD при p_intra={p_within}, p_inter={p_between}, p_input={p_input}',
    fontsize=18
)
plt.savefig('results/PSD_with_SNR_readable.pdf', format='pdf')
plt.savefig('results/PSD_with_SNR_readable.svg', format='svg')
plt.show()
