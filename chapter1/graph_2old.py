import os
import pickle
import pandas as pd
from scipy.stats import mannwhitneyu

# 1. Две группы условий
groups = {
    'one_random_without_boost': {
        'one\ncluster\nwithout\nboost':    'results_ext_test1_one_cluster_bez_boost',
        'random\nneighbors\nwithout\nboost': 'results_ext_test1_random_neighbors_bez_boost',
    },
    'one_random_with_boost': {
        'one\ncluster\nwith\nboost':    'results_ext_test1_one_cluster_s_boost',
        'random\nneighbors\nwith\nboost': 'results_ext_test1_random_neighbors_s_boost',
    },
}

# 2. Общие параметры анализа
p_intra      = '0.15'
p_inter_vals = [0.01, 0.04, 0.07, 0.1]
p_input_perc = [10, 15, 20]
measure_names = [
    "degree",
    "betweenness",
    "closeness",
    "random",
    "eigenvector",
    "percolation",
    "harmonic",
]

def analyze_and_save(cond_dict, label, out_dir):
    # создаём папку для результатов группы, если её ещё нет
    os.makedirs(out_dir, exist_ok=True)

    # 1) загрузка данных для двух условий
    loaded = {}
    cond1, cond2 = cond_dict.keys()
    for cond, dpath in cond_dict.items():
        print(dpath)
        with open(os.path.join(dpath, 'spike_counts_second_cluster.pickle'), 'rb') as f:
            loaded[cond] = pickle.load(f)

    # 2) сбор и тестирование
    records = []
    for measure in measure_names:
        data1 = loaded[cond1][1000][measure][p_intra]
        data2 = loaded[cond2][1000][measure][p_intra]
        for p_inter in p_inter_vals:
            arrs1 = data1[p_inter]
            arrs2 = data2[p_inter]
            for idx in range(len(arrs1)):
                print(measure, p_inter, idx)
                print(arrs1[idx])
                print(arrs2[idx])
                u, pval = mannwhitneyu(arrs1[idx], arrs2[idx], alternative='two-sided')
                records.append({
                    'measure':     measure,
                    'p_inter':     p_inter,
                    'p_input_%':   p_input_perc[idx],
                    'U_statistic': u,
                    'p_value':     round(pval, 5)
                })

    df_all = pd.DataFrame(records)
    df_sig = df_all[df_all['p_value'] < 0.05]

    # 3) сохранение
    path_all = os.path.join(out_dir, f'mannwhitney_{label}_all.csv')
    path_sig = os.path.join(out_dir, f'mannwhitney_{label}_significant.csv')
    df_all.to_csv(path_all, index=False)
    df_sig.to_csv(path_sig, index=False)

    print(f'[{label}] результаты сохранены в папку "{out_dir}":')
    print(f'  • все результаты — {path_all}')
    print(f'  • значимые p<0.05 — {path_sig}')

# 4. Запуск анализа для всех групп
base_results_dir = 'results'
for label, conds in groups.items():
    out_dir = os.path.join(base_results_dir, label)
    analyze_and_save(conds, label, out_dir)
