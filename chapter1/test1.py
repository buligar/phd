import os
import pandas as pd

# Список папок с результатами
folders = [
    'results_ext_test1_one_cluster_bez_boost',
    'results_ext_test1_one_cluster_s_boost',
    'results_ext_test1_random_neighbors_bez_boost',
    'results_ext_test1_random_neighbors_s_boost',
    'results_ext_test1_top_neighbors_bez_boost',
    'results_ext_test1_top_neighbors_s_boost'
]

results = []

for folder in folders:
    path = os.path.join(folder, 'avg_tests_avg_spikes_5000ms.csv')
    if not os.path.isfile(path):
        continue
    df = pd.read_csv(path)
    
    # 1) Центральность с максимальным mean_spikes и её std_spikes
    grouped = df.groupby('measure_name')['mean_spikes'].max()
    top_cent = grouped.idxmax()
    top_mean = grouped.max()
    idx_top = df[(df['measure_name'] == top_cent) & (df['mean_spikes'] == top_mean)].index[0]
    std_top = df.loc[idx_top, 'std_spikes']
    
    # 2) Максимум для random и его std_spikes
    rand_df = df[df['measure_name'] == 'random']
    max_rand = rand_df['mean_spikes'].max()
    idx_rand = rand_df['mean_spikes'].idxmax()
    std_rand = df.loc[idx_rand, 'std_spikes']
    
    # 3) Центральность с максимальным mean_spikes при p_between == 0.07 и её std_spikes
    p07_df = df[df['p_between'] == 0.07]
    if not p07_df.empty:
        max_p07 = p07_df['mean_spikes'].max()
        idx_p07 = p07_df['mean_spikes'].idxmax()
        cent_p07 = df.loc[idx_p07, 'measure_name']
        std_p07 = df.loc[idx_p07, 'std_spikes']
    else:
        cent_p07, max_p07, std_p07 = (None, None, None)
    
    # 4) Значения для random при p_between == 0.07
    rand_p07_df = p07_df[p07_df['measure_name'] == 'random']
    if not rand_p07_df.empty:
        mean_rand_p07 = rand_p07_df['mean_spikes'].iloc[0]
        idx_rand_p07 = rand_p07_df.index[0]
        std_rand_p07 = df.loc[idx_rand_p07, 'std_spikes']
    else:
        mean_rand_p07, std_rand_p07 = (None, None)
    
    results.append({
        'folder': folder,
        'top_centrality': top_cent,
        'top_mean_spikes': top_mean,
        'std_top_spikes': std_top,
        'max_random_spikes': max_rand,
        'std_random_spikes': std_rand,
        'p07_centrality': cent_p07,
        'p07_max_spikes': max_p07,
        'std_p07_spikes': std_p07,
        'random_p07_mean_spikes': mean_rand_p07,
        'std_random_p07_spikes': std_rand_p07
    })

# Сборка DataFrame
results_df = pd.DataFrame(results)

# Округление до 4 десятичных знаков
num_cols = results_df.select_dtypes(include=['float64', 'int64']).columns
results_df[num_cols] = results_df[num_cols].round(4)

# Сохранение с десятичной запятой
output_file = 'results/summary_results.csv'
results_df.to_csv(output_file, index=False, sep=';', decimal=',')

print(f"Сводный файл сохранён: {output_file}")

# Загрузка итоговой таблицы
df = pd.read_csv('results/summary_results.csv', sep=';', decimal=',')

# Извлечение нужных значений top_mean_spikes
oc_bez = df.loc[
    df['folder']=='results_ext_test1_one_cluster_bez_boost',
    'top_mean_spikes'
].iloc[0]
rn_bez = df.loc[
    df['folder']=='results_ext_test1_random_neighbors_bez_boost',
    'top_mean_spikes'
].iloc[0]

oc_s   = df.loc[
    df['folder']=='results_ext_test1_one_cluster_s_boost',
    'top_mean_spikes'
].iloc[0]
rn_s   = df.loc[
    df['folder']=='results_ext_test1_random_neighbors_s_boost',
    'top_mean_spikes'
].iloc[0]

# Расчёт reduction=(one_cluster - random_neighbors)/one_cluster*100
reduction_bez = (oc_bez - rn_bez) / oc_bez * 100
reduction_s   = (oc_s   - rn_s)   / oc_s   * 100

# Округляем до двух знаков и выводим
print(f"Reduction (one_cluster vs random_neighbors) без буста: {reduction_bez:.2f} %")
print(f"Reduction (one_cluster vs random_neighbors) с бустом:    {reduction_s:.2f} %")


# 2. Извлечение p07_max_spikes для каждого случая:
oc_bez_p07 = df.loc[
    df['folder'] == 'results_ext_test1_one_cluster_bez_boost',
    'p07_max_spikes'
].item()
rn_bez_p07 = df.loc[
    df['folder'] == 'results_ext_test1_random_neighbors_bez_boost',
    'p07_max_spikes'
].item()

oc_s_p07 = df.loc[
    df['folder'] == 'results_ext_test1_one_cluster_s_boost',
    'p07_max_spikes'
].item()
rn_s_p07 = df.loc[
    df['folder'] == 'results_ext_test1_random_neighbors_s_boost',
    'p07_max_spikes'
].item()

# 3. Расчёт reduction = (one_cluster - random_neighbors) / one_cluster * 100
reduction_bez_p07 = (oc_bez_p07 - rn_bez_p07) / oc_bez_p07 * 100
reduction_s_p07   = (oc_s_p07   - rn_s_p07)   / oc_s_p07   * 100

# 4. Вывод (округление до двух десятичных знаков)
print(f"Reduction p_between=0.07, без буста: {reduction_bez_p07:.2f} %")
print(f"Reduction p_between=0.07, с бустом:   {reduction_s_p07:.2f} %")