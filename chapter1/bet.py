import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Построим граф Krackhardt kite
G = nx.krackhardt_kite_graph()
nodes = sorted(G.nodes())
n = len(nodes)

# Вычислим betweenness-centrality
bc = nx.betweenness_centrality(G)

# Для каждой вершины v создадим маску пар (i,j),
# для которых хотя бы один кратчайший путь i->j проходит через v
masks = {}
for v in nodes:
    mask = np.zeros((n, n), dtype=int)
    for i, u in enumerate(nodes):
        for j, w in enumerate(nodes):
            if i >= j: 
                continue
            # проверяем все кратчайшие пути
            for path in nx.all_shortest_paths(G, u, w):
                if v in path[1:-1]:
                    mask[i, j] = mask[j, i] = 1
                    break
    masks[v] = mask

# Визуализация: 2 строки × 5 столбцов, увеличенный шрифт
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
cmap = 'viridis'
title_fs = 16   # размер шрифта заголовка
tick_fs = 12    # размер шрифта отметок

for idx, v in enumerate(nodes):
    ax = axes[idx // 5, idx % 5]
    im = ax.imshow(masks[v], cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_title(f"u = {v} (BC={bc[v]:.3f})", fontsize=title_fs)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(nodes, fontsize=tick_fs)
    ax.set_yticklabels(nodes, fontsize=tick_fs)

plt.tight_layout()
plt.show()
