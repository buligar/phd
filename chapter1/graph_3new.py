import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

def plot_3d_with_significance(
    stats_csvs,
    pvals_csv,
    labels,
    colormap_names=None,
    alphas_meas=None,
    alphas_rand=None,
    p_input_values=None,
    p_between_values=None,
    max_rate=10,
    I0_value_filter=1000,
    output_pdf="results/combined_with_stars.pdf",
    output_svg="results/combined_with_stars.svg",
    measure_names=None
):
    # 0) сопоставление measure → заголовок subplot
    title_map = {
        "closeness":    "Степень близости\n против случайных узлов /\n closeness vs random",
        "degree":       "Степень вершины\n против случайных узлов /\n degree vs random",
        "betweenness":  "Степень посредничества\n против случайных узлов /\n betweenness vs random",
        "eigenvector":  "Степень влиятельности\n против случайных узлов /\n eigenvector vs random",
        "harmonic":     "Гармоническая центральность\n против случайных узлов /\n harmonic vs random",
        "percolation":   "Центральность просачивания\n против случайных узлов /\n percolation vs random",
    }

    # 1) p-values
    pvals = pd.read_csv(pvals_csv)
    pvals = pvals.rename(columns={"p_inter": "p_between", "p_input_%": "p_input"})
    pvals["p_input"] /= 100.0

    # 2) stats
    stats_dfs = []
    for f in stats_csvs:
        df = pd.read_csv(f)
        if "I0_value" in df.columns:
            df = df[df["I0_value"] == I0_value_filter]
        stats_dfs.append(df)

    # 3) выбор мер (без внутреннего 'random')
    all_measures = sorted(stats_dfs[0]["measure_name"].unique())
    measures = [m for m in all_measures if m != "random"]
    if measure_names:
        measures = [m for m in measure_names if m in measures]
    measures = measures[:6]

    # baseline random
    df_rand_base = stats_dfs[0][stats_dfs[0]["measure_name"] == "random"]

    # 4) сборка фигуры
    fig = plt.figure(figsize=(16, 14))
    zmax = max_rate * 1.05
    norm = mpl.colors.Normalize(vmin=0, vmax=max_rate)
    meas_cmap = plt.get_cmap(colormap_names[0] if colormap_names else "magma")
    rand_cmap = plt.get_cmap(colormap_names[1] if colormap_names else "spring")

    for idx, measure in enumerate(measures):
        ax = fig.add_subplot(2, 3, idx+1, projection="3d")
        # здесь вместо простого measure ставим наше человеко-машино-читаемое название
        title = title_map.get(measure,
                              f"{measure} vs random")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r"$p_{input}$", fontsize=16)
        ax.set_ylabel(r"$p_{inter}$", fontsize=16)
        ax.set_zlim(0, zmax)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        if p_input_values is not None:
            ax.set_xticks(p_input_values)
        if p_between_values is not None:
            ax.set_yticks(p_between_values)

        # — далее ваш существующий код отрисовки поверхностей и звёздочек —
        df_p = pvals[pvals["measure"] == measure]
        pm_list = []
        for df_stats in stats_dfs:
            df_m0 = df_stats[df_stats["measure_name"] == measure]
            if "p_within" in df_m0.columns:
                pw0 = sorted(df_m0["p_within"].unique(), key=float)[0]
                df_m0 = df_m0[df_m0["p_within"] == pw0]
            inp = sorted(set(df_m0["p_input"]) & set(df_p["p_input"]), key=float)
            bet = sorted(set(df_m0["p_between"]) & set(df_p["p_between"]), key=float)
            pm = df_m0.pivot_table("mean_spikes", "p_input", "p_between",
                                    aggfunc="mean").loc[inp, bet].values
            pm_list.append(pm)
        pm_stack = np.stack(pm_list, axis=0)
        argmax_k = np.argmax(pm_stack, axis=0)

        for k, (df_stats, label) in enumerate(zip(stats_dfs, labels)):
            df_m = df_stats[df_stats["measure_name"] == measure]
            if "p_within" in df_m.columns:
                pw0 = sorted(df_m["p_within"].unique(), key=float)[0]
                df_m = df_m[df_m["p_within"] == pw0]
                df_rand = df_rand_base[df_rand_base["p_within"] == pw0]
            else:
                df_rand = df_rand_base

            inp = sorted(set(df_m["p_input"]) & set(df_rand["p_input"]) &
                         set(df_p["p_input"]), key=float)
            bet = sorted(set(df_m["p_between"]) & set(df_rand["p_between"]) &
                         set(df_p["p_between"]), key=float)

            pm = pm_stack[k]
            sm = df_m.pivot_table("std_spikes", "p_input", "p_between",
                                   aggfunc="mean").loc[inp, bet].values
            pr = df_rand.pivot_table("mean_spikes", "p_input", "p_between",
                                     aggfunc="mean").loc[inp, bet].values
            sr = df_rand.pivot_table("std_spikes", "p_input", "p_between",
                                     aggfunc="mean").loc[inp, bet].values
            pv = df_p.pivot_table("p_value", "p_input", "p_between",
                                   aggfunc="mean").loc[inp, bet].values

            X, Y = np.meshgrid(np.array(inp, float), np.array(bet, float),
                               indexing="ij")

            ax.plot_surface(
                X, Y, pm, cmap=meas_cmap, edgecolor="none",
                alpha=(alphas_meas[k] if alphas_meas else 1), norm=norm
            )
            if k == 0:
                ax.plot_surface(
                    X, Y, pr, cmap=rand_cmap, edgecolor="none",
                    alpha=(alphas_rand[k] if alphas_rand else 1), norm=norm
                )

            # отрисовка ошибок и звёздочек p<0.0005
            for i in range(pm.shape[0]):
                for j in range(pm.shape[1]):
                    x0, y0 = X[i, j], Y[i, j]
                    z0, dz = pm[i, j], sm[i, j]
                    z1, dr = pr[i, j], sr[i, j]
                    ax.plot([x0, x0], [y0, y0], [z0 - dz, z0 + dz], 'k-', lw=1)
                    ax.plot([x0, x0], [y0, y0], [z1 - dr, z1 + dr], 'k-', lw=1)
                    pval = pv[i, j]
                    stars = "***" if pval < 0.0005 else ""
                    if stars and k == argmax_k[i, j]:
                        star_color = ('k' if (inp[i]==0.1 or bet[j]==0.1) else 'w')
                        ax.text(
                            x0, y0, z0 + max_rate*0.02,
                            stars, ha="center", va="bottom",
                            fontsize=20, color=star_color,
                            zorder=99, clip_on=False
                        )

    # 5) остальные элементы (colorbar, легенда и сохранение) без изменений
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95)

    # ——— горизонтальная colorbar для avg_spikes (magma) ———
    sm_avg = mpl.cm.ScalarMappable(cmap=meas_cmap, norm=norm)
    sm_avg.set_array([])
    cax_avg = fig.add_axes([0.10, 0.05, 0.35, 0.02])  # [left, bottom, width, height]
    cbar_avg = fig.colorbar(sm_avg, cax=cax_avg, orientation='horizontal')
    cbar_avg.set_label(r"$\overline{\mathrm{spikes}}$", fontsize=20)
    cbar_avg.ax.tick_params(labelsize=16)

    # ——— горизонтальная colorbar для random (spring) ———
    sm_rand = mpl.cm.ScalarMappable(cmap=rand_cmap, norm=norm)
    sm_rand.set_array([])
    cax_rand = fig.add_axes([0.55, 0.05, 0.35, 0.02])
    cbar_rand = fig.colorbar(sm_rand, cax=cax_rand, orientation='horizontal')
    cbar_rand.set_label(r"$\overline{\mathrm{spikes}}_{\mathrm{random}}$", fontsize=20)
    cbar_rand.ax.tick_params(labelsize=16)

    # единая легенда p-value над avg_spikes
    cax_avg.text(
        0.5, 1.3,
        "*** p < 0.0005",
        ha='center', va='bottom',
        transform=cax_avg.transAxes,
        fontsize=20
    )

    plt.savefig(output_pdf, format="pdf")
    plt.savefig(output_svg, format="svg")
    plt.show()
if __name__ == "__main__":
    # Создаём папку «results», если необходимо
    if not os.path.exists("results"):
        os.mkdir("results")

    plot_3d_with_significance(
        stats_csvs=[
            "results_ext_test1/avg_tests_avg_spikes_5000ms.csv",
        ],
        pvals_csv="results/metrics_vs_random/mannwhitney_metrics_vs_random_all.csv",
        labels=["one_cluster_s_boost"],
        colormap_names=["magma", "spring"],
        alphas_meas=[1],
        alphas_rand=[0.4],
        p_input_values=np.arange(0.1, 0.21, 0.05),
        p_between_values=np.arange(0.01, 0.11, 0.03),
        max_rate=10,
        I0_value_filter=1000,
        output_pdf="results/3D_one_s_boost.pdf",
        output_svg="results/3D_one_s_boost.svg"
    )
