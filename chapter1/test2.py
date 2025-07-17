#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для вычисления усреднённой матрицы пересечений топ-N узлов
по нескольким прогонкам эксперимента и визуализации результата.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

def compute_mean_matrices(
    p_within: float,
    p_between_list: list,
    p_input_list: list,
    measure_names: list,
    mode: str,
    test_indices: range
) -> np.ndarray:
    """
    Загружает данные топ-N узлов для каждого прогона и параметров,
    вычисляет матрицу пересечений (в %) и усредняет по прогонкам.

    Возвращает массив shape (B, I, M, M), где
      B = len(p_between_list),
      I = len(p_input_list),
      M = len(measure_names).
    """
    T = len(test_indices)
    B = len(p_between_list)
    I = len(p_input_list)
    M = len(measure_names)

    # Тензор для хранения всех матриц: [прогон, между, вход, мера_i, мера_j]
    all_matrices = np.zeros((T, B, I, M, M), dtype=float)

    for t, test_i in enumerate(test_indices):
        for i, p_between in enumerate(p_between_list):
            for j, p_input in enumerate(p_input_list):
                # Формируем пути к файлам и загружаем множества узлов
                node_sets = {}
                for m in measure_names:
                    path = (
                        f"results_ext_test1/top_nodes/"
                        f"top_nodes_{p_within:.2f}_{p_between}_{p_input}_"
                        f"{m}_{mode}_{test_i}.csv"
                    )
                    df = pd.read_csv(path, header=None)
                    # Предполагаем, что индексы узлов во втором столбце
                    node_sets[m] = set(df.iloc[:, 1].astype(str))

                # Составляем матрицу пересечений для данного прогона
                mat = np.zeros((M, M), dtype=float)
                for u, m1 in enumerate(measure_names):
                    S1 = node_sets[m1]
                    size_S1 = len(S1)
                    for v, m2 in enumerate(measure_names):
                        if size_S1 > 0:
                            S2 = node_sets[m2]
                            intersection_size = len(S1 & S2)
                            mat[u, v] = intersection_size / size_S1 * 100.0
                        else:
                            mat[u, v] = 0.0
                all_matrices[t, i, j] = mat

    # Усредняем по прогонкам (ось 0)
    mean_matrices = np.mean(all_matrices, axis=0)
    return mean_matrices


def plot_mean_matrices(
    mean_matrices: np.ndarray,
    p_between_list: list,
    p_input_list: list,
    measure_names: list,
    output_path: str
) -> None:
    """
    Строит сетку субплотов усреднённых матриц пересечений и сохраняет в SVG.
    """
    B, I, M, _ = mean_matrices.shape

    fig, axes = plt.subplots(
        nrows=B,
        ncols=I,
        figsize=(16, 16),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    # Отрисовка каждого субплота
    for i in range(B):
        for j in range(I):
            ax = axes[i, j]
            mat = mean_matrices[i, j]
            im = ax.imshow(mat, vmin=0, vmax=100, aspect='auto')

            # Текстовые аннотации
            for yi, xi in product(range(M), range(M)):
                ax.text(
                    xi, yi,
                    f"{mat[yi, xi]:.1f}%",
                    ha='center', va='center',
                    fontsize=10, color='white'
                )

            ax.set_title(
                fr"$p_{{inter}}={p_between_list[i]:.2f},\ p_{{input}}={p_input_list[j]:.2f}$",
                fontsize=16
            )
            ax.set_xticks(range(M))
            ax.set_xticklabels(measure_names, rotation=90, fontsize=16)
            ax.set_yticks(range(M))
            ax.set_yticklabels(measure_names, fontsize=16)

    # Общий заголовок и colorbar
    fig.suptitle(
        "Усреднённый процент пересечений топ-N узлов",
        fontsize=20
    )
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8)
    cbar.set_label('% пересечений', fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    # Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format='svg')
    plt.show()


if __name__ == "__main__":
    # Параметры эксперимента
    p_within       = 0.15
    p_between_list = [0.01, 0.04, 0.07, 0.1]
    p_input_list   = [0.1, 0.15, 0.2]
    measure_names  = [
        'betweenness', 'degree', 'closeness',
        'random', 'eigenvector', 'harmonic',
        'percolation'
    ]
    mode           = 'one_cluster_bez_boost'
    test_indices   = range(1)  # например, 0,1,2,3,4

    # Вычисление усреднённых матриц
    mean_mats = compute_mean_matrices(
        p_within,
        p_between_list,
        p_input_list,
        measure_names,
        mode,
        test_indices
    )

    # Визуализация и сохранение
    output_svg = "results/coincidence_mean.svg"
    plot_mean_matrices(
        mean_mats,
        p_between_list,
        p_input_list,
        measure_names,
        output_svg
    )
