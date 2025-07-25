import numpy as np
import pandas as pd
from collections import deque

# Правильный Krackhardt kite graph
nodes = list(range(10))
adj = {
    0: [1, 2, 3, 5],
    1: [0, 3, 4, 6],
    2: [0, 3, 5],
    3: [0, 1, 2, 4, 5, 6],
    4: [1, 3, 6],
    5: [0, 2, 3, 6, 7],
    6: [1, 3, 4, 5, 7],
    7: [5, 6, 8],
    8: [7, 9],
    9: [8]
}
n = len(nodes)

# 1. Degree centrality
degree = {v: len(adj[v]) / (n - 1) for v in nodes}

# 2. BFS для расстояний
def bfs(start):
    dist = {v: np.inf for v in nodes}
    dist[start] = 0
    Q = deque([start])
    while Q:
        u = Q.popleft()
        for w in adj[u]:
            if dist[w] == np.inf:
                dist[w] = dist[u] + 1
                Q.append(w)
    return dist

# 3. Closeness и Harmonic centrality
closeness = {}
harmonic = {}
for v in nodes:
    dist = bfs(v)
    total = sum(d for u, d in dist.items() if u != v)
    closeness[v] = (n - 1) / total
    harmonic[v]  = sum(1.0 / d for u, d in dist.items() if u != v)

# 4. Betweenness centrality (Brandes)
betweenness = dict.fromkeys(nodes, 0.0)
for s in nodes:
    S, P = [], {v: [] for v in nodes}
    sigma = dict.fromkeys(nodes, 0)
    sigma[s] = 1
    dist  = dict.fromkeys(nodes, -1) 
    dist[s]  = 0
    Q = deque([s])
    while Q:
        v = Q.popleft(); S.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                Q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v] 
                P[w].append(v)
    delta = dict.fromkeys(nodes, 0.0)
    while S:
        w = S.pop()
        for v in P[w]:
            delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
        if w != s:
            betweenness[w] += delta[w]

# Нормализация для неориентированного графа
factor = 1 / ((n - 1) * (n - 2))
for v in nodes:
    betweenness[v] *= factor

# 5. Eigenvector centrality (power iteration)
A = np.zeros((n, n))
for i in nodes:
    for j in adj[i]:
        A[i, j] = 1
x = np.ones(n)
for _ in range(1000):
    x_new = A.dot(x)
    x_new /= np.linalg.norm(x_new)
    if np.allclose(x, x_new, atol=1e-6):
        break
    x = x_new
eigenvector = {v: x[v] for v in nodes}

# 6. Percolation centrality (равные состояния → совпадает с betweenness)
percolation = betweenness.copy()

# Сбор в DataFrame и вывод
df_manual = pd.DataFrame({
    'degree': degree,
    'betweenness': betweenness,
    'closeness': closeness,
    'eigenvector': eigenvector,
    'percolation': percolation,
    'harmonic': harmonic
}).sort_index().round(6)

print(df_manual)


import networkx as nx

G = nx.krackhardt_kite_graph()
states = {v: 1 for v in G.nodes()}

df_nx = pd.DataFrame({
    'degree':       nx.degree_centrality(G),
    'betweenness':  nx.betweenness_centrality(G),
    'closeness':    nx.closeness_centrality(G),
    'eigenvector':  nx.eigenvector_centrality(G),
    'percolation':  nx.percolation_centrality(G, states=states),
    'harmonic':     nx.harmonic_centrality(G)
}).sort_index().round(6)

print(df_nx)
