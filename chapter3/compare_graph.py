import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================

XLSX_PATH = "results_PC-EC_PC-SC_10tests_8metrics_4x2/summary_metrics_mean_std.xlsx"   # поменяй на свой файл
SHEET_NAME = 0

OUT_DIR = Path("ijbc_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600
SHOW = False

# Если mean_phase_deg нет в xlsx, будет построен sync_score
USE_SYNC_SCORE_IF_NO_PHASE = True


# =============================================================================
# IJBC STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "savefig.dpi": DPI,
    "figure.dpi": 150,
})


# =============================================================================
# METRICS
# =============================================================================

def get_plot_metrics(df):
    """
    RMSE metrics are fixed in the last row.
    If mean_phase_deg exists, use it.
    Otherwise use sync_score = mean of amp/freq/PLV.
    """

    metrics = [
        ("sim_time_s", "Simulation time (s)", False),
        ("sim_python_peak_mem_mb", "Python peak memory (MB)", False),

        ("mean_amp_corr", "Amplitude correlation", True),
        ("mean_freq_corr", "Frequency correlation", True),

        ("mean_plv", "Phase-locking value", True),
    ]

    if "mean_phase_deg" in df.columns or "mean_phase_deg_mean" in df.columns:
        metrics.append(("mean_phase_deg", "Mean phase difference (deg)", False))
    elif USE_SYNC_SCORE_IF_NO_PHASE:
        metrics.append(("sync_score", "Synchronization score", True))
    else:
        metrics.append(("mean_phase_deg", "Mean phase difference (deg)", False))

    # last row
    metrics.extend([
        ("rmse_ref_o1", r"RMSE(reference, $o_1$)", False),
        ("rmse_o1_g", r"RMSE($o_1$, $g$)", False),
    ])

    return metrics


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_xlsx(path, sheet_name=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    if "N_sens" not in df.columns:
        raise ValueError(f"Missing column: N_sens. Available columns: {list(df.columns)}")

    if "arch" not in df.columns:
        raise ValueError(f"Missing column: arch. Available columns: {list(df.columns)}")

    if "signal" not in df.columns:
        df["signal"] = "lorenz"

    df["arch"] = df["arch"].astype(str)

    return df


def add_sync_score_if_needed(df):
    needed = ["mean_amp_corr", "mean_freq_corr", "mean_plv"]

    if all(c in df.columns for c in needed):
        df["sync_score"] = df[needed].mean(axis=1)

    elif all(f"{c}_mean" in df.columns for c in needed):
        df["sync_score_mean"] = df[[f"{c}_mean" for c in needed]].mean(axis=1)

        std_cols = [f"{c}_std" for c in needed]
        if all(c in df.columns for c in std_cols):
            # approximate propagated std for average of 3 metrics
            df["sync_score_std"] = np.sqrt(
                df[std_cols[0]] ** 2 +
                df[std_cols[1]] ** 2 +
                df[std_cols[2]] ** 2
            ) / 3.0
        else:
            df["sync_score_std"] = 0.0

    return df


def prepare_mean_std(df, metrics):
    """
    Supports:
    1) raw table:
       signal, N_sens, arch, test_id, metric...

    2) already aggregated table:
       signal, N_sens, arch, metric_mean, metric_std...

    3) single-run table:
       signal, N_sens, arch, metric...
    """

    df = add_sync_score_if_needed(df)

    metric_names = [m[0] for m in metrics]
    group_cols = ["signal", "N_sens", "arch"]

    has_agg = any(f"{m}_mean" in df.columns for m in metric_names)

    if has_agg:
        rows = []

        for _, r in df.iterrows():
            row = {
                "signal": r["signal"],
                "N_sens": r["N_sens"],
                "arch": r["arch"],
            }

            for metric in metric_names:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"

                if mean_col in df.columns:
                    row[f"{metric}_mean"] = r[mean_col]
                elif metric in df.columns:
                    row[f"{metric}_mean"] = r[metric]
                else:
                    row[f"{metric}_mean"] = np.nan

                if std_col in df.columns:
                    row[f"{metric}_std"] = r[std_col]
                else:
                    row[f"{metric}_std"] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    rows = []

    for (signal, n_sens, arch), g in df.groupby(group_cols):
        row = {
            "signal": signal,
            "N_sens": n_sens,
            "arch": arch,
        }

        for metric in metric_names:
            if metric in g.columns:
                values = pd.to_numeric(g[metric], errors="coerce")
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if values.count() > 1 else 0.0
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.965,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def prettify_arch_name(name):
    name = str(name)
    if name.upper() == "PC-EC":
        return "PC-EC"
    if name.upper() == "PC-SC":
        return "PC-SC"
    return name


def plot_ijbc_4x2(df_ms, metrics, out_dir):
    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]

    for signal_name in sorted(df_ms["signal"].dropna().unique()):
        df_sig = df_ms[df_ms["signal"] == signal_name].copy()

        fig, axes = plt.subplots(4, 2, figsize=(15.5, 18.5))
        axes = axes.ravel()

        for ax, panel_label, (metric, ylabel, ylim01) in zip(axes, panel_labels, metrics):
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            for arch in sorted(df_sig["arch"].dropna().unique()):
                g = df_sig[df_sig["arch"] == arch].sort_values("N_sens")

                if mean_col not in g.columns:
                    continue

                x = g["N_sens"].to_numpy(dtype=float)
                y = g[mean_col].to_numpy(dtype=float)

                if std_col in g.columns:
                    y_std = g[std_col].to_numpy(dtype=float)
                else:
                    y_std = np.zeros_like(y)

                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                y_std = y_std[valid]

                if x.size == 0:
                    continue

                label = prettify_arch_name(arch)

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.4,
                    markersize=6.5,
                    label=label,
                )

                if np.isfinite(y_std).any() and np.nanmax(y_std) > 0:
                    ax.fill_between(
                        x,
                        y - y_std,
                        y + y_std,
                        alpha=0.18,
                        linewidth=0,
                    )

            add_panel_label(ax, panel_label)

            ax.set_title(ylabel, pad=8)
            ax.set_xlabel(r"$N_{\mathrm{sens}}$")
            ax.set_ylabel(ylabel)

            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.minorticks_on()

            if ylim01:
                ax.set_ylim(-0.05, 1.05)

            # For error/time/memory plots, start from zero when possible
            # if not ylim01:
            #     ymin, ymax = ax.get_ylim()
            #     if ymin > 0:
            #         ax.set_ylim(0, ymax * 1.05)

        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels),
                frameon=False,
                bbox_to_anchor=(0.5, 0.993),
                handlelength=2.5,
            )


        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])

        out_png = out_dir / f"ijbc_metrics_4x2_{signal_name}.png"
        out_pdf = out_dir / f"ijbc_metrics_4x2_{signal_name}.pdf"
        out_svg = out_dir / f"ijbc_metrics_4x2_{signal_name}.svg"

        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")

        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_svg}")

        if SHOW:
            plt.show()

        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = load_xlsx(XLSX_PATH, SHEET_NAME)

    print("Loaded columns:")
    for c in df.columns:
        print(" -", c)

    metrics = get_plot_metrics(df)

    print("\nMetrics for plotting:")
    for m, label, _ in metrics:
        print(f" - {m}: {label}")

    df_ms = prepare_mean_std(df, metrics)

    used_table_path = OUT_DIR / "mean_std_used_for_ijbc_plot.xlsx"
    df_ms.to_excel(used_table_path, index=False)
    print(f"\nSaved processed table: {used_table_path}")

    missing = []
    for metric, _, _ in metrics:
        if f"{metric}_mean" not in df_ms.columns or df_ms[f"{metric}_mean"].isna().all():
            missing.append(metric)

    if missing:
        print("\nWARNING: missing metrics:")
        for m in missing:
            print(" -", m)

    plot_ijbc_4x2(df_ms, metrics, OUT_DIR)


if __name__ == "__main__":
    main()