import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# 1. Параметры анализа
data_dir       = 'results_ext_test1'
pickle_name    = 'spike_counts_second_cluster.pickle'
p_intra        = '0.15'
p_inter_vals   = [0.01, 0.04, 0.07, 0.1]
p_input_perc   = [10, 15, 20]
measure_names  = [
    "degree",
    "betweenness",
    "closeness",
    "random",
    "eigenvector",
    "percolation",
    "harmonic",
]

# 2. Загрузка данных
with open(os.path.join(data_dir, pickle_name), 'rb') as f:
    loaded_all = pickle.load(f)
# Предполагается, что ключ 1000 присутствует во вложенном словаре
data = loaded_all[1000]

# 3. Подготовка списка пар для сравнения
metrics_to_compare = [m for m in measure_names if m != 'random']

# 4. Расчёт теста Манна–Уитни
records = []
for measure in metrics_to_compare:
    dist_m = data[measure][p_intra]
    dist_r = data['random'][p_intra]
    for p_inter in p_inter_vals:
        arrs_m = dist_m[p_inter]
        arrs_r = dist_r[p_inter]
        for idx, inp in enumerate(p_input_perc):
            x = arrs_m[idx]
            y = arrs_r[idx]
            # 4.1. Проверка на «избыток» нулей
            if (np.count_nonzero(x == 0) > x.size / 2) or (np.count_nonzero(y == 0) > y.size / 2):
                U_stat, p_val = 1, 1.0
            else:
                U_stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
            records.append({
                'measure':           f'{measure}',
                'p_inter':        p_inter,
                'p_input_%':      inp,
                'U_statistic':    U_stat,
                'p_value':        p_val
            })

df = pd.DataFrame(records)

# 5. Сохранение результатов
out_dir = os.path.join('results', 'metrics_vs_random')
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, 'mannwhitney_metrics_vs_random_all.csv'), index=False)
df[df['p_value'] < 0.05]\
  .to_csv(os.path.join(out_dir, 'mannwhitney_metrics_vs_random_significant.csv'),
          index=False)

print(f'Готово: все результаты сохранены в "{out_dir}"')
