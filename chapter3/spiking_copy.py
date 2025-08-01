import numpy as np
import os.path
import pickle
import brian2 as b
from struct import unpack
from brian2 import *
from tensorflow.keras.datasets import mnist
from spectral_connectivity import Multitaper, Connectivity
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import mutual_info_score
import networkx as nx
import pandas as pd
prefs.codegen.target = 'cython'
start_scope()

MNIST_data_path = ''

os.makedirs(f'weights', exist_ok=True)


def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

def save_connections():
    conn = XeAe
    connListSparse = list(zip(conn.i[:], conn.j[:], conn.w[:]))  # делаем список из zip
    connArray = np.array(connListSparse)                         # преобразуем в numpy-массив
    print(connArray.shape)
    np.save(os.path.join(data_path, 'weights', 'XeAe.npy'), connArray)

def save_theta():
    np.save(data_path + 'weights/theta_A', Ae.theta)

def normalize_weights():
    len_source = len(XeAe.source)
    len_target = len(XeAe.target)
    connection = np.zeros((len_source, len_target))
    connection[XeAe.i, XeAe.j] = XeAe.w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis=0)
    colFactors = 78. / colSums
    for j in range(n_e):
        temp_conn[:, j] *= colFactors[j]
    XeAe.w = temp_conn[XeAe.i, XeAe.j]



def load_mnist_keras():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = {'x': x_train, 'y': y_train, 'rows': x_train.shape[1], 'cols': x_train.shape[2]}
    test_data = {'x': x_test, 'y': y_test, 'rows': x_test.shape[1], 'cols': x_test.shape[2]}
    return train_data, test_data

training, testing = load_mnist_keras()
print("Размер обучающей выборки:", training['x'].shape)
print("Размер тестовой выборки:", testing['x'].shape)

test_mode = True
np.random.seed(0)
data_path = './'
num_examples = 100 if test_mode else 200

n_input = 784
n_e = 400
n_i = 400
single_example_time = 0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre = 0.0001
nu_ee_post = 0.01
wmax_ee = 1.0

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'

neuron_eqs_e = '''
dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
I_synE = ge * nS * -v : amp
I_synI = gi * nS * (-100.*mV-v) : amp
dge/dt = -ge/(1.0*ms) : 1
dgi/dt = -gi/(2.0*ms) : 1'''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
I_synE = ge * nS * -v : amp
I_synI = gi * nS * (-85.*mV-v) : amp
dge/dt = -ge/(1.0*ms) : 1
dgi/dt = -gi/(2.0*ms) : 1'''

eqs_stdp_ee = '''
post2before : 1
dpre/dt = -pre/(tc_pre_ee) : 1 (event-driven)
dpost1/dt = -post1/(tc_post_1_ee) : 1 (event-driven)
dpost2/dt = -post2/(tc_post_2_ee) : 1 (event-driven)'''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

Ae = b.NeuronGroup(n_e, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, method='euler')
Ai = b.NeuronGroup(n_i, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, method='euler')

Ae.v = v_rest_e - 40. * b.mV
Ai.v = v_rest_i - 40. * b.mV
if test_mode:
    Ae.theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    Ae.theta = np.ones((n_e)) * 20.0*b.mV

weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
AeAi = b.Synapses(Ae, Ai, model='w : 1', on_pre='ge_post += w')
AeAi.connect(True)
AeAi.w = weightMatrix[AeAi.i, AeAi.j]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
AiAe = b.Synapses(Ai, Ae, model='w : 1', on_pre='gi_post += w')
AiAe.connect(True)
AiAe.w = weightMatrix[AiAe.i, AiAe.j]


Xe = b.PoissonGroup(n_input, 0*Hz)

Ae_voltage_monitor = b.StateMonitor(Ae, 'v', record=True, dt=0.01*second)
Ai_voltage_monitor = b.StateMonitor(Ai, 'v', record=True, dt=0.01*second)
Xe_spike_monitor = b.SpikeMonitor(Xe)
Ae_spike_monitor = b.SpikeMonitor(Ae)
Ai_spike_monitor = b.SpikeMonitor(Ai)
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''
if not test_mode:
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee


def enforce_target_connectivity(G, cluster_sizes, p_intra, p_inter):
    """
    Корректирует граф G таким образом, чтобы количество внутрикластерных
    и межкластерных рёбер соответствовало p_intra и p_inter.

    Рёбра удаляются сначала по наименьшей значимости: для каждого ребра
    считаем score = cent[u] + cent[v], где cent — degree centrality.
    Удаляем те ребра, у которых score наименьший.
    """
    # 1. Подготовка
    n = sum(cluster_sizes)
    # метки кластеров
    labels = np.empty(n, dtype=int)
    idx = 0
    for cl, sz in enumerate(cluster_sizes):
        labels[idx:idx+sz] = cl
        idx += sz

    # 2. Подсчёт целевых чисел рёбер
    intra_possible = sum(sz*(sz-1)//2 for sz in cluster_sizes)
    inter_possible = 0
    for i in range(len(cluster_sizes)):
        for j in range(i+1, len(cluster_sizes)):
            inter_possible += cluster_sizes[i]*cluster_sizes[j]

    target_intra = int(round(p_intra * intra_possible))
    target_inter = int(round(p_inter * inter_possible))

    # 3. Вычисление текущего набора рёбер
    intra_edges = []
    inter_edges = []
    for u, v in G.edges():
        if labels[u] == labels[v]:
            intra_edges.append((u, v))
        else:
            # учитываем только u<v, но обход G.edges даёт каждое только раз
            inter_edges.append((u, v))

    curr_intra = len(intra_edges)
    curr_inter = len(inter_edges)

    # 4. Оценка значимости рёбер через degree centrality
    cent = nx.degree_centrality(G)
    def edge_score(edge):
        u, v = edge
        return cent[u] + cent[v]

    # 5. Удаление лишних внутрикластерных рёбер
    if curr_intra > target_intra:
        # сортируем по возрастанию значимости
        sorted_intra = sorted(intra_edges, key=edge_score)
        remove_count = curr_intra - target_intra
        for edge in sorted_intra[:remove_count]:
            if G.has_edge(*edge):
                G.remove_edge(*edge)

    # 6. Удаление лишних межкластерных рёбер
    if curr_inter > target_inter:
        sorted_inter = sorted(inter_edges, key=edge_score)
        remove_count = curr_inter - target_inter
        for edge in sorted_inter[:remove_count]:
            if G.has_edge(*edge):
                G.remove_edge(*edge)

    return G

def _build_base_sbm(n, cluster_sizes, p_intra, p_inter):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    clusters = []
    start = 0
    rng = np.random.default_rng(42)
    for sz in cluster_sizes:
        C = list(range(start, start+sz))
        clusters.append(C)
        start += sz
    for C in clusters:
        for i in C:
            for j in C:
                if j>i and rng.random() < p_intra:
                    G.add_edge(i, j)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for u in clusters[i]:
                for v in clusters[j]:
                    if rng.random() < p_inter:
                        G.add_edge(u, v)
    return G, clusters


def minmax_normalize(centrality_dict):
    """
    Выполняет min–max нормализацию словаря центральностей, 
    чтобы все значения лежали в диапазоне [0,1].
    Если все значения в centrality_dict одинаковые, то всем присваивается 1.0.
    """
    if not centrality_dict:
        return centrality_dict  # пустой словарь

    values = list(centrality_dict.values())
    min_val = min(values)
    max_val = max(values)
    if np.isclose(max_val, min_val):
        # Если все значения одинаковые, присвоим 1.0 всем
        for k in centrality_dict:
            centrality_dict[k] = 1.0
    else:
        diff = max_val - min_val
        for k in centrality_dict:
            centrality_dict[k] = (centrality_dict[k] - min_val) / diff
    return centrality_dict


def get_centrality(G, clusters, ctype, norm, cluster_type):
    if ctype == "degree":
        cent = nx.degree_centrality(G)
    elif ctype == "betweenness":
        cent = nx.betweenness_centrality(G)
    elif ctype == "eigenvector":
        cent = nx.eigenvector_centrality(G, max_iter=1000)
    elif ctype == "percolation":
        cent = nx.percolation_centrality(G)
    elif ctype == "closeness":
        cent = nx.closeness_centrality(G)
    elif ctype == "harmonic":
        cent = nx.harmonic_centrality(G)
    elif ctype == "local_clustering":
        cent = nx.clustering(G)
    elif ctype == "random":
        rng = np.random.default_rng()
        if cluster_type == 'cluster12':
            cent = {node: rng.random() for node in G.nodes()}
        elif cluster_type == 'cluster1':
            target_cluster = clusters[0]
            subG = G.subgraph(target_cluster)
            cent = {node: rng.random() for node in subG.nodes()}
    else:
        raise ValueError(f"Unsupported centrality type: {ctype}")

    if norm:
        cent = minmax_normalize(cent)
    return cent

def generate_sbm_with_high_centrality(
    n, cluster_sizes, p_intra, p_inter,
    target_cluster_index, proportion_high_centrality=0.2,
    centrality_type="degree", boost_factor=2.0,
    mode='one_cluster_s_boost'
):
    G, clusters = _build_base_sbm(n, cluster_sizes, p_intra, p_inter)
    G_before = G.copy()
    scores = get_centrality(G_before, clusters, centrality_type, norm=False, cluster_type='cluster12')
    # выбираем топ‑узлы в целевом кластере
    C = clusters[target_cluster_index]
    k = int(np.ceil(proportion_high_centrality * len(C)))
    sorted_C = sorted(C, key=lambda u: scores[u], reverse=True)
    top_nodes = sorted_C[:k]
    # буст внешних рёбер топ‑узлов
    G_mod = G_before.copy()
    max_score = max(scores[u] for u in top_nodes) or 1.0
    for u in top_nodes:
        nbr_ext = [v for v in range(n) if v not in clusters[target_cluster_index] and not G_mod.has_edge(u, v)]
        norm_s = scores[u] / max_score
        k_ext = sum(1 for v in range(n) if G_mod.has_edge(u, v) and v not in clusters[target_cluster_index])
        delta_k = int(np.ceil((boost_factor - 1.0) * norm_s * max(1, k_ext)))
        chosen = np.random.choice(nbr_ext, min(len(nbr_ext), delta_k), replace=False)
        for v in chosen:
            G_mod.add_edge(u, v)
    G_after = enforce_target_connectivity(G_mod, cluster_sizes, p_intra, p_inter)
    return G_before, G_after

# ========== генерируем SBM с бустом ==========
n = n_input + n_e
cluster_sizes = [n_input, n_e]
p_intra, p_inter = 0.15, 0.6

G_before, G_after = generate_sbm_with_high_centrality(
    n, cluster_sizes, p_intra, p_inter,
    target_cluster_index=1,      # увеличиваем центральность блокa Ae
    proportion_high_centrality=0.1,
    centrality_type="degree",
    boost_factor=2,
    mode='one_cluster_s_boost'
)

# ========== извлекаем матрицу смежности вход→Ae ==========
A = nx.to_numpy_array(G_before, dtype=int)
# A  = nx.to_numpy_array(G_after, dtype=int)

A_block = A[:n_input, n_input:]
plt.figure(figsize=(6,6))
plt.imshow(A_block, cmap='Greys', interpolation='nearest', aspect='auto')
plt.title('SBM: вход→Ae (построено по A[:n_input, n_input:])')
plt.xlabel('Номер нейрона Ae')
plt.ylabel('Номер входного нейрона Xe')
plt.colorbar(label='1 = связь, 0 = нет')
plt.tight_layout()
plt.show()

pre_all, post_all = np.where(A > 0)
mask = (pre_all < n_input) & (post_all >= n_input)
i_xe = pre_all[mask]
j_ae = post_all[mask] - n_input

XeAe = b.Synapses(Xe, Ae, model=model, on_pre=pre, on_post=post)
# XeAe.connect(True)
XeAe.connect(i=i_xe, j=j_ae)

minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay
XeAe.delay = 'minDelay + rand() * deltaDelay'
XeAe.w = weightMatrix[XeAe.i, XeAe.j]

w_init = np.zeros((n_input, n_e))
w_init[XeAe.i[:], XeAe.j[:]] = XeAe.w[:]


net = Network(Ae, Ai, Xe, AeAi, AiAe, XeAe, Ae_spike_monitor,Ae_voltage_monitor,Ai_voltage_monitor, Xe_spike_monitor)

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples
Xe.rates = 0 * Hz
net.run(0*second)
j = 0
result_monitor = np.zeros((num_examples, n_e)) if test_mode else None
previous_Ae_count = np.zeros(n_e, dtype=int)
previous_Ai_count = np.zeros(n_i, dtype=int)
previous_Xe_count = np.zeros(n_input, dtype=int)



while j < int(num_examples):
    print(j)
    print(training['y'][j])
    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. * input_intensity
    else:
        normalize_weights()
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. * input_intensity
    Xe.rates = rate * Hz
    net.run(single_example_time, report='text')
    current_spike_count = np.asarray(Ae_spike_monitor.count[:]) - previous_spike_count
    previous_spike_count = np.copy(Ae_spike_monitor.count[:])

    if np.sum(current_spike_count) < 5:
        input_intensity += 10
        Xe.rates = 0 * Hz
        net.run(resting_time)
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print('runs done:', j, 'of', int(num_examples))
        Xe.rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1
        

        graph = False
        if graph:
            t_ob = Ae_voltage_monitor.t / b.second   

            # Мембранные потенциалы
            t_e = Ae_voltage_monitor.t / ms          # (n_times,)
            V_e = Ae_voltage_monitor.v / mV          # (n_neurons, n_times) 
            t_i = Ae_voltage_monitor.t / b.ms        # время в миллисекундах, форма (n_times,)
            V_i = Ae_voltage_monitor.v / b.mV        # потенциал в милливольтах, форма (n_ne
            

            spike_times1 = Xe_spike_monitor.t / ms    # времена спайков
            spike_indices1 = Xe_spike_monitor.i       # индексы нейронов
            # Растер спайков
            spike_times2 = Ae_spike_monitor.t / ms    # времена спайков
            spike_indices2 = Ae_spike_monitor.i       # индексы нейронов

            spike_times3 = Ai_spike_monitor.t / ms    # времена спайков
            spike_indices3 = Ai_spike_monitor.i       # индексы нейронов

            
            current_Xe_count = Xe_spike_monitor.count[:]                 
            new_Xe_spikes = (current_Xe_count - previous_Xe_count) > 0    
            previous_Xe_count = current_Xe_count.copy()
            img_Xe = new_Xe_spikes.reshape((28, 28))  # 28×28

            current_Ae_count = Ae_spike_monitor.count[:]                 
            new_Ae_spikes = (current_Ae_count - previous_Ae_count) > 0    
            previous_Ae_count = current_Ae_count.copy()
            img_Ae = new_Ae_spikes.reshape((20, 20))  # 28×28

            # Спайки на выходе Ae
            current_Ai_count = Ai_spike_monitor.count[:]                   
            new_Ai_spikes = (current_Ai_count - previous_Ai_count) > 0     
            previous_Ai_count = current_Ai_count.copy()
            img_Ai = new_Ai_spikes.reshape((20, 20))  # 28×28, если n_e=784

            # 2) Рисуем единый figure с 2×2 субплотами
            fig, axes = plt.subplots(2, 3, figsize=(12, 10))
            
            ax = axes[0, 0]
            ax.scatter(spike_times1, spike_indices1, marker='|')
            ax.set_xlabel('Время (мс)')
            ax.set_ylabel('№ нейрона')
            ax.set_title('Растер спайков (Xe)')

            # — Верхний левый: растер спайков
            ax = axes[0, 1]
            ax.scatter(spike_times2, spike_indices2, marker='|')
            ax.set_xlabel('Время (мс)')
            ax.set_ylabel('№ нейрона')
            ax.set_title('Растер спайков (Ae)')

            # — Верхний правый: мембранные потенциалы (первые 5 нейронов)
            ax = axes[0, 2]
            ax.scatter(spike_times3, spike_indices3, marker='|')
            ax.set_xlabel('Время (мс)')
            ax.set_ylabel('№ нейрона')
            ax.set_title('Растер спайков (Ai)')

            ax = axes[1, 0]
            ax.imshow(img_Xe, cmap='hot', interpolation='nearest', aspect='equal')
            ax.set_title('Спайки на выходе (Xe)')
            ax.axis('off')

            # — Нижний левый: образ входных спайков 28×28
            ax = axes[1, 1]
            ax.imshow(img_Ae, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title('Спайки на входе (Ae)')
            ax.axis('off')

            # — Нижний правый: образ выходных спайков 20×20
            ax = axes[1, 2]
            ax.imshow(img_Ai, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title('Спайки на выходе (Ai)')
            ax.axis('off')

            t_spikes = np.array(Ae_spike_monitor.t / ms)  # [ms]
            i_spikes = np.array(Ae_spike_monitor.i)       # [0…783]

            # Общее максимальное время спайков
            t_max = t_spikes.max() if t_spikes.size else single_example_time/ms

            # Задать число кадров и FPS
            num_frames = 20
            fps = 10

            # Разбить интервал [0, t_max] на num_frames отрезков
            frame_edges = np.linspace(0, t_max, num_frames + 1)

            # Подготовка списка изображений
            frames = []
            for k in range(num_frames):
                t_start = frame_edges[k]
                t_end   = frame_edges[k+1]
                # выбор спайков в текущем временном окне
                mask = (t_spikes > t_start) & (t_spikes <= t_end)
                inds = i_spikes[mask]
                # формирование бинарного изображения 28×28
                img = np.zeros((20, 20), dtype=np.uint8)
                for idx in inds:
                    row, col = divmod(idx, 20)
                    img[row, col] = 255  # белый пиксель
                frames.append(img)

            # --- Создание и сохранение GIF ---
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.axis('off')
            # первый кадр
            im = ax2.imshow(frames[0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')

            def update(frame_index):
                im.set_data(frames[frame_index])
                return (im,)

            ani_img = FuncAnimation(fig2, update, frames=len(frames),
                                    blit=True, interval=1000/fps)

            # Папка для анимаций
            os.makedirs(os.path.join(data_path, 'animations'), exist_ok=True)
            gif_path_img = os.path.join(data_path, 'animations', 'Ae_spikes_image.gif')
            ani_img.save(gif_path_img, writer=PillowWriter(fps=fps))

            print(f"Анимация 28×28-изображений спайков Ae сохранена в: {gif_path_img}")

            t_spikes2 = np.array(Xe_spike_monitor.t / ms)  # [ms]
            i_spikes2 = np.array(Xe_spike_monitor.i)       # [0…783]

            # Общее максимальное время спайков
            t_max2 = t_spikes2.max() if t_spikes2.size else single_example_time/ms

            # Задать число кадров и FPS
            num_frames = 20
            fps = 10

            # Разбить интервал [0, t_max] на num_frames отрезков
            frame_edges2 = np.linspace(0, t_max2, num_frames + 1)

            # Подготовка списка изображений
            frames2 = []
            for k in range(num_frames):
                t_start = frame_edges2[k]
                t_end   = frame_edges2[k+1]
                # выбор спайков в текущем временном окне
                mask2 = (t_spikes2 > t_start) & (t_spikes2 <= t_end)
                inds = i_spikes2[mask2]
                # формирование бинарного изображения 28×28
                img2 = np.zeros((28, 28), dtype=np.uint8)
                for idx in inds:
                    row, col = divmod(idx, 28)
                    img2[row, col] = 255  # белый пиксель
                frames2.append(img2)

            # --- Создание и сохранение GIF ---
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.axis('off')
            # первый кадр
            im2 = ax3.imshow(frames2[0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')

            def update2(frame_index2):
                im2.set_data(frames2[frame_index2])
                return (im2,)

            ani_img2 = FuncAnimation(fig3, update2, frames=len(frames2),
                                    blit=True, interval=1000/fps)

            # Папка для анимаций
            os.makedirs(os.path.join(data_path, 'animations'), exist_ok=True)
            gif_path_img2 = os.path.join(data_path, 'animations', 'Xe_spikes_image.gif')
            ani_img2.save(gif_path_img2, writer=PillowWriter(fps=fps))

            print(f"Анимация 28×28-изображений спайков Ae сохранена в: {gif_path_img2}")
    

    generate_gif_spike = False  # флаг сохранения GIF-анимации

    if generate_gif_spike:
        t_spikes = np.array(Xe_spike_monitor.t / ms)
        i_spikes = np.array(Xe_spike_monitor.i)
        t_max = t_spikes.max() if t_spikes.size else 1.0
        num_frames = 20
        fps = 20

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, t_max)
        ax.set_ylim(-1, n_input)
        ax.set_xlabel('Время, мс')
        ax.set_ylabel('Индекс нейрона')
        scatter = ax.scatter([], [], marker='|')

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            return scatter,

        def update(frame_time):
            mask = t_spikes <= frame_time
            data = np.column_stack((t_spikes[mask], i_spikes[mask]))
            scatter.set_offsets(data)
            return scatter,

        frame_times = np.linspace(0, t_max, num_frames)
        ani = FuncAnimation(fig, update, frames=frame_times,
                            init_func=init, blit=True, interval=30)

        os.makedirs(os.path.join(data_path, 'animations'), exist_ok=True)
        gif_path = os.path.join(data_path, 'animations', 'Xe_spikes.gif')
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"Анимация растра спайков Xe сохранена в: {gif_path}")


        t_spikes2 = np.array(Ae_spike_monitor.t / ms)
        i_spikes2 = np.array(Ae_spike_monitor.i)
        t_max2 = t_spikes2.max() if t_spikes2.size else 1.0
        num_frames = 20
        fps = 20

        fig2, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, t_max)
        ax.set_ylim(-1, n_input)
        ax.set_xlabel('Время, мс')
        ax.set_ylabel('Индекс нейрона')
        scatter2 = ax.scatter([], [], marker='|')

        def init2():
            scatter2.set_offsets(np.empty((0, 2)))
            return scatter2,

        def update2(frame_time):
            mask2 = t_spikes2 <= frame_time
            data2 = np.column_stack((t_spikes2[mask2], i_spikes2[mask2]))
            scatter2.set_offsets(data2)
            return scatter2,

        frame_times2 = np.linspace(0, t_max2, num_frames)
        ani2 = FuncAnimation(fig2, update2, frames=frame_times2,
                            init_func=init2, blit=True, interval=30)

        os.makedirs(os.path.join(data_path, 'animations'), exist_ok=True)
        gif_path2 = os.path.join(data_path, 'animations', 'Ae_spikes.gif')
        ani2.save(gif_path2, writer=PillowWriter(fps=fps))
        print(f"Анимация растра спайков Xe сохранена в: {gif_path2}")
        plt.show()

W_after = np.zeros((400, 400))
W_after[AeAi.i[:], AeAi.j[:]] = AeAi.w[:]
W_after[AiAe.i[:], AiAe.j[:]] = AiAe.w[:]

plt.imshow(W_after)
plt.show()

os.makedirs(os.path.join(data_path,'activity'), exist_ok=True)
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)


w_final = np.zeros((n_input, n_e))
w_final[XeAe.i[:], XeAe.j[:]] = XeAe.w[:]

delta = w_final - w_init
plt.figure(figsize=(6,5))
plt.imshow(delta, aspect='auto', cmap='bwr',
           vmin=-np.max(np.abs(delta)), vmax=np.max(np.abs(delta)))
plt.colorbar(label='Δw (Xe→Ae)')
plt.xlabel('post‑neuron index (Ae)')
plt.ylabel('pre‑neuron index (Xe)')
plt.title('Изменение весов Xe→Ae после STDP')
plt.tight_layout()
plt.show()

# 2) Создаём DataFrame и сохраняем
coords = np.argwhere(delta > 0)
df = pd.DataFrame(coords, columns=['pre', 'post'])
csv_path = os.path.join(data_path, 'positive_synapses.csv')
df.to_csv(csv_path, index=False)
print(f"Сохранено {len(df)} синапсов с Δw>0 в файл {csv_path}")

# -----------------------
# 1) Параметры и пути
# -----------------------
data_path = './activity/'
training_suffix = '100'
testing_suffix  = '100'

n_e     = 400

# -----------------------
# 2) Загрузка spike‑count векторов и меток
# -----------------------
train_counts = np.load(os.path.join(data_path, f'resultPopVecs{training_suffix}.npy'))
train_labels = np.load(os.path.join(data_path, f'inputNumbers{training_suffix}.npy')).flatten()

test_counts  = np.load(os.path.join(data_path, f'resultPopVecs{testing_suffix}.npy'))
test_labels  = np.load(os.path.join(data_path, f'inputNumbers{testing_suffix}.npy')).flatten()


num_test = test_counts.shape[0]

# -----------------------
# 3) Функция для назначения нейронов классам
# -----------------------
def get_new_assignments(counts, labels):
    """
    Для каждого нейрона находим цифру, при которой он
    в среднем выстреливает сильнее всего в тренировке.
    """
    n_e = counts.shape[1]
    assignments = np.zeros(n_e, dtype=int)
    # посчитаем среднюю активность каждого нейрона для каждой цифры
    mean_rates = np.zeros((10, n_e))
    for digit in range(10):
        mask = (labels == digit)
        if mask.any():
            mean_rates[digit] = counts[mask].mean(axis=0)
    # назначаем нейрон цифре с макс средней
    assignments = mean_rates.argmax(axis=0)
    return assignments

# -----------------------
# 4) Получаем assignments по тренировочному набору
# -----------------------
assignments = get_new_assignments(train_counts, train_labels)
print(f"Assignments array shape: {assignments.shape}")

# -----------------------
# 5) Декодирование теста: прямая сумма + argmax
# -----------------------
preds = np.zeros(num_test, dtype=int)
for i in range(num_test):
    vec = test_counts[i]           # spike‑count вектор для примера i
    sums = np.zeros(10)
    for neuron_idx, digit in enumerate(assignments):
        sums[digit] += vec[neuron_idx]
    preds[i] = sums.argmax()



# ───────────────────────────────────────────────────────────────────
# 5) Декодирование теста: прямая сумма + argmax
# ───────────────────────────────────────────────────────────────────
preds = np.zeros(num_test, dtype=int)
score_mat = np.zeros((num_test, 10))        # сохраняем сами суммы — пригодится
for i in range(num_test):
    vec = test_counts[i]
    sums = np.zeros(10)
    for neuron_idx, digit in enumerate(assignments):
        sums[digit] += vec[neuron_idx]
    preds[i]    = sums.argmax()
    score_mat[i] = sums

# ───────────────────────────────────────────────────────────────────
# 6) METRICS: точность, MI, ошибка, матрица ошибок, F1, SNR, энтропии
# ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support
)
from scipy.stats import entropy

true = test_labels
accuracy      = np.mean(preds == true)
balanced_acc  = balanced_accuracy_score(true, preds)
error_rate    = 1.0 - accuracy
kappa         = cohen_kappa_score(true, preds)

# Mutual information
I_bits = mutual_info_score(true, preds) / np.log(2)

# Энтропии
H_true  = entropy(np.bincount(true, minlength=10)/len(true), base=2)
H_pred  = entropy(np.bincount(preds, minlength=10)/len(preds), base=2)
cm      = confusion_matrix(true, preds, labels=np.arange(10))
p_joint = cm / cm.sum()
H_joint = entropy(p_joint.flatten(), base=2)
H_cond  = H_joint - H_pred            # H(true | pred)
redund  = H_pred / H_true if H_true else np.nan

# SNR в «простом» определении: (µ₀−µ₁)² / (σ₀²+σ₁²)/2  между двумя наиболее частыми цифрами
top2 = np.argsort(np.bincount(true))[-2:]
mu   = [score_mat[true == d].mean()  for d in top2]
var  = [score_mat[true == d].var()   for d in top2]
snr  = (mu[1] - mu[0])**2 / (0.5*(var[0] + var[1])) if var[0]+var[1] else np.nan

# Precision/Recall/F1 (macro и по классам)
prec, rec, f1, _ = precision_recall_fscore_support(true, preds, labels=np.arange(10), zero_division=0)
report = classification_report(true, preds, digits=3)

# ───── вывод ───────────────────────────────────────────────────────
print("\n=== MNIST TEST METRICS ===")
print(f"Accuracy (overall)      : {accuracy*100:.2f}%")
print(f"Balanced accuracy       : {balanced_acc*100:.2f}%")
print(f"Error rate              : {error_rate*100:.2f}%")
print(f"Cohen’s κ               : {kappa:.3f}")
print(f"Mutual information I    : {I_bits:.3f} bits")
print(f"Entropy true H(Y)       : {H_true:.3f} bits")
print(f"Entropy pred H(Ŷ)       : {H_pred:.3f} bits")
print(f"Conditional H(Y|Ŷ)      : {H_cond:.3f} bits")
print(f"Redundancy H(Ŷ)/H(Y)    : {redund:.3f}")
print(f"SNR (top‑2 classes)     : {snr:.3f}")
print("\nPer‑class precision/recall/F1:\n")
print(report)

# Матрица ошибок
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues'); plt.colorbar()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion matrix')
plt.xticks(np.arange(10)); plt.yticks(np.arange(10))
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=8)
plt.tight_layout(); plt.show()
