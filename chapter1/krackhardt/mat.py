import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Построение графа Krackhardt kite и всех матриц
G = nx.krackhardt_kite_graph()
nodes = sorted(G.nodes())
n = len(nodes)

A = nx.to_numpy_array(G, nodelist=nodes, dtype=int)      # смежность
deg = A.sum(axis=1).astype(int)
Dg = np.diag(deg)                                        # диагональная матрица степеней
L = Dg - A                                               # лапласиан
lengths = dict(nx.all_pairs_shortest_path_length(G))     # расстояния
Dmat = np.zeros((n, n), dtype=int)
for i, u in enumerate(nodes):
    for j, v in enumerate(nodes):
        Dmat[i, j] = lengths[u][v]

# Единая цветовая карта
cmap = 'viridis'

# Визуализация: 2×2 сетка
fig, axes = plt.subplots(1, 3, figsize=(12, 10))

# 1) Adjacency
ax = axes[0]
im0 = ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect='equal')
ax.set_title("Матрица смежности (A)")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(nodes); ax.set_yticklabels(nodes)
# Цветовая шкала для A
cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

# 2) Degree
ax = axes[1]
im1 = ax.imshow(Dg, cmap=cmap, aspect='equal')
ax.set_title("Матрица степени (D)")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(nodes); ax.set_yticklabels(nodes)
# Цветовая шкала для D
cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label("Степень")


# 4) Distance heatmap
ax = axes[2]
im3 = ax.imshow(Dmat, cmap=cmap, aspect='equal')
ax.set_title("Матрица расстояний H")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(nodes); ax.set_yticklabels(nodes)
cbar3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
cbar3.set_label("Расстояние")

plt.tight_layout()
plt.show()
