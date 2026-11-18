#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot separation, scheduling, and human evaluation results from summary tables."""


import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


METHOD_COLORS = {
    "EBAMA": "#4C78A8",
    "EBAMA+Sep": "#54A24B",
    "EBAMA+Adaptive": "#F58518",
    "SA-EBAMA": "#E45756",
}

PAIR_COLORS = {
    "EBAMA": "#4C78A8",
    "SA-EBAMA": "#E45756",
}

SCHED_COLORS = {
    "Fixed-25": "#4C78A8",
    "Early-10": "#54A24B",
    "Uniform-10": "#B279A2",
    "Sparse-10": "#F58518",
    "Sparse-10+trigger": "#72B7B2",
    "Final": "#E45756",
}

SCHED_MARKERS = {
    "Fixed-25": "o",
    "Early-10": "s",
    "Uniform-10": "D",
    "Sparse-10": "^",
    "Sparse-10+trigger": "v",
    "Final": "*",
}


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def savefig(fig, out_dir: Path, name: str, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)
    fig.savefig(out_dir / f"{name}.pdf")
    print(f"[Saved] {out_dir / f'{name}.png'}")
    print(f"[Saved] {out_dir / f'{name}.pdf'}")
    plt.close(fig)


def clean_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def find_col(df: pd.DataFrame, candidates, contains=None):
    for c in candidates:
        if c in df.columns:
            return c
    if contains:
        for c in df.columns:
            low = c.lower()
            if all(s.lower() in low for s in contains):
                return c
    raise KeyError(f"Cannot find column. candidates={candidates}, columns={list(df.columns)}")


def draw_fig1(table4: pd.DataFrame, out_dir: Path, dpi: int):
    labels_map = {
        "ebama": "EBAMA",
        "ebama_sep": "EBAMA+Sep",
        "ebama_adaptive": "EBAMA+Adaptive",
        "sa_ebama": "SA-EBAMA",
        "EBAMA": "EBAMA",
        "EBAMA+Sep": "EBAMA+Sep",
        "EBAMA+Adaptive": "EBAMA+Adaptive",
        "SA-EBAMA": "SA-EBAMA",
    }
    order = ["EBAMA", "EBAMA+Sep", "EBAMA+Adaptive", "SA-EBAMA"]

    method_col = find_col(table4, ["method", "Method"])
    oor_col = find_col(table4, ["OOR↓", "OOR", "oor"], contains=["oor"])
    upd_col = find_col(table4, ["Update Steps↓", "Update Steps", "update_steps"], contains=["update"])
    time_col = find_col(table4, ["Time/Image↓", "Time/Image", "time_image"], contains=["time"])

    df = table4.copy()
    df["label"] = df[method_col].map(lambda x: labels_map.get(str(x), str(x)))
    df = df.set_index("label").loc[order].reset_index()

    colors = [METHOD_COLORS[x] for x in order]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05))

    metrics = [
        (oor_col, "Object Overlap Ratio", "Object Attention Overlap Ratio", (0.22, 0.305), "{:.4f}"),
        (upd_col, "Number of Latent Update Steps", "Latent Update Steps", (0, 30), "{:.1f}"),
        (time_col, "Inference Time per Image (s)", "Inference Time per Image", (0, 8), "{:.3f}"),
    ]

    for ax, (col, ylabel, title, ylim, fmt) in zip(axes, metrics):
        vals = df[col].astype(float).values
        x = np.arange(len(order))
        bars = ax.bar(x, vals, width=0.66, color=colors, edgecolor="none", linewidth=0.0)
        bars[-1].set_linewidth(1.4)

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=22, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, pad=8, fontweight="bold", fontsize=10)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=9)

        for idx, (b, v) in enumerate(zip(bars, vals)):
            off = (ylim[1] - ylim[0]) * 0.025
            is_sa = order[idx] == "SA-EBAMA"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + off,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_sa else "normal",
            )

        clean_axes(ax)

    fig.tight_layout(w_pad=2.0)
    savefig(fig, out_dir, "fig1_ablation_efficiency_sci", dpi)


def parse_mask_method(method: str):
    method = str(method)
    m = re.match(r".*pct(\d+)_temp(\d+).*", method)
    if not m:
        return np.nan, np.nan
    return int(m.group(1)) / 100.0, int(m.group(2)) / 100.0


def prepare_mask_matrix(mask: pd.DataFrame):
    if {"sep_percentile", "sep_temp"}.issubset(set(mask.columns)):
        mdf = mask.copy()
        mdf["pct"] = mdf["sep_percentile"].astype(float)
        mdf["temp"] = mdf["sep_temp"].astype(float)
    else:
        method_col = find_col(mask, ["method", "Method"])
        mdf = mask.copy()
        mdf[["pct", "temp"]] = mdf[method_col].apply(lambda x: pd.Series(parse_mask_method(x)))

    overlap_col = find_col(mdf, ["mean_overlap", "Mean overlap", "mean overlap"], contains=["overlap"])
    pcts = sorted(mdf["pct"].dropna().unique())
    temps = sorted(mdf["temp"].dropna().unique())

    mat = np.array([
        [mdf[(mdf["pct"] == p) & (mdf["temp"] == t)][overlap_col].astype(float).iloc[0] for t in temps]
        for p in pcts
    ])
    return pcts, temps, mat


def prepare_schedule(schedule: pd.DataFrame):
    sched_map = {
        "B_fixed25": "Fixed-25",
        "B_early10": "Early-10",
        "B_uniform10": "Uniform-10",
        "B_sparse10": "Sparse-10",
        "B_sparse10_trigger": "Sparse-10+trigger",
        "B_final": "Final",
        "Fixed-25": "Fixed-25",
        "Early-10": "Early-10",
        "Uniform-10": "Uniform-10",
        "Sparse-10": "Sparse-10",
        "Sparse-10+trigger": "Sparse-10+trigger",
        "Final": "Final",
    }
    order = ["Fixed-25", "Early-10", "Uniform-10", "Sparse-10", "Sparse-10+trigger", "Final"]

    method_col = find_col(schedule, ["method", "Method"])
    time_col = find_col(schedule, ["time_image", "Time/Image", "Time/Image↓"], contains=["time"])
    overlap_col = find_col(schedule, ["mean_overlap", "Mean overlap"], contains=["overlap"])
    update_col = find_col(schedule, ["update_steps", "Update Steps", "Update Steps↓"], contains=["update"])

    sdf = schedule.copy()
    sdf["label"] = sdf[method_col].map(lambda x: sched_map.get(str(x), str(x)))
    sdf = sdf.set_index("label").loc[order].reset_index()
    sdf["time_image"] = sdf[time_col].astype(float)
    sdf["mean_overlap"] = sdf[overlap_col].astype(float)
    sdf["update_steps"] = sdf[update_col].astype(float)
    return sdf


def scatter_point(ax, row, size=None):
    label = row["label"]
    if size is None:
        size = 62 if row["update_steps"] <= 10.1 else 120
        if label == "Final":
            size = 135
    ax.scatter(
        row["time_image"], row["mean_overlap"],
        s=size,
        color=SCHED_COLORS[label],
        marker=SCHED_MARKERS[label],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.96,
        zorder=5,
    )


def draw_fig2(mask: pd.DataFrame, schedule: pd.DataFrame, out_dir: Path, dpi: int):
    """
    Fig. 2: parameter sensitivity and scheduler analysis.

    Final paper-oriented layout:
    - left: compact heatmap;
    - right-top: global trade-off scatter with all schedules visible;
    - right-bottom: zoomed 10-step scatter;
    - no in-plot text labels or leader lines in scatter plots;
    - a larger shared schedule legend is placed above the global scatter panel.
    """
    pcts, temps, mat = prepare_mask_matrix(mask)
    sdf = prepare_schedule(schedule)

    fig = plt.figure(figsize=(10.6, 5.02))

    # Manual layout: left heatmap; right upper/lower scatter panels.
    # Compared with v8, the lower scatter panel is moved upward to reduce excessive
    # vertical blank space, while the legend is lifted above panel (b) to avoid
    # overlapping the title.
    ax1 = fig.add_axes([0.065, 0.225, 0.335, 0.610])
    cax = fig.add_axes([0.408, 0.225, 0.016, 0.610])
    ax2 = fig.add_axes([0.545, 0.635, 0.415, 0.235])  # global scatter
    ax3 = fig.add_axes([0.545, 0.150, 0.415, 0.275])  # zoomed scatter

    # (a) Heatmap.
    im = ax1.imshow(mat, cmap="YlGnBu", aspect="equal", vmin=mat.min(), vmax=mat.max())
    ax1.set_xticks(np.arange(len(temps)))
    ax1.set_xticklabels([f"{t:.2f}" for t in temps])
    ax1.set_yticks(np.arange(len(pcts)))
    ax1.set_yticklabels([f"{p:.2f}" for p in pcts])
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Separation Percentile")
    ax1.set_title("(a) Soft-Mask Parameter Sensitivity", pad=8, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 0.30 else "black"
            ax1.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8, color=color)

    if 0.70 in pcts and 0.15 in temps:
        formal_i = pcts.index(0.70)
        formal_j = temps.index(0.15)
        ax1.add_patch(Rectangle((formal_j - 0.5, formal_i - 0.5), 1.0, 1.0,
                                fill=False, edgecolor="#D62728", linewidth=2.0))

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean Overlap")
    ax1.grid(False)

    # Scheduler data.
    fixed = sdf[sdf["label"] == "Fixed-25"].iloc[0]
    cluster = sdf[sdf["label"] != "Fixed-25"].copy()

    # Shared legend handles. Keep legend outside scatter data regions.
    legend_handles, legend_labels = [], []
    for _, r in sdf.iterrows():
        label = r["label"]
        size = 112 if label == "Fixed-25" else (104 if label == "Final" else 84)
        h = ax2.scatter([], [], s=size, color=SCHED_COLORS[label], marker=SCHED_MARKERS[label],
                        edgecolor="black", linewidth=0.70, alpha=0.96)
        legend_handles.append(h)
        legend_labels.append(label)

    # (b) Global trade-off scatter.
    # Slight display offsets only in the global overview to make the clustered
    # 10-step schedules visible. The zoomed panel below uses exact data positions.
    global_offsets = {
        "Fixed-25": (0.000,  0.0000),
        "Early-10": (-0.085, -0.00025),
        "Uniform-10": (-0.070,  0.00035),
        "Sparse-10": (-0.020,  0.00012),
        "Sparse-10+trigger": (0.055,  0.00000),
        "Final": (0.110, -0.00012),
    }
    for _, r in sdf.iterrows():
        label = r["label"]
        dx, dy = global_offsets[label]
        size = 112 if label == "Fixed-25" else (104 if label == "Final" else 84)
        ax2.scatter(r["time_image"] + dx, r["mean_overlap"] + dy,
                    s=size, color=SCHED_COLORS[label], marker=SCHED_MARKERS[label],
                    edgecolor="black", linewidth=0.70, alpha=0.96, zorder=5)

    ax2.set_xlabel("Inference Time per Image (s)")
    ax2.set_ylabel("Mean Overlap")
    ax2.set_title("(b) Global Trade-off", pad=7, fontweight="bold")
    ax2.set_xlim(5.43, 7.36)
    ax2.set_ylim(0.242, 0.262)
    clean_axes(ax2)

    # Larger shared legend, placed higher than v8 so it does not overlap panel (b)'s title.
    ax2.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.24),
        ncol=3,
        handletextpad=0.40,
        columnspacing=1.05,
        fontsize=8.0,
        borderaxespad=0.0,
        markerscale=1.05,
    )

    # (c) Zoomed 10-step schedules.
    # Exact data positions, no text labels and no leader lines.
    for _, r in cluster.iterrows():
        label = r["label"]
        size = 82 if label != "Final" else 126
        ax3.scatter(r["time_image"], r["mean_overlap"],
                    s=size, color=SCHED_COLORS[label], marker=SCHED_MARKERS[label],
                    edgecolor="black", linewidth=0.70, alpha=0.96, zorder=5)

    ax3.set_xlabel("Inference Time per Image (s)")
    ax3.set_ylabel("Mean Overlap")
    ax3.set_title("(c) Zoomed 10-Step Schedules", pad=7, fontweight="bold")
    ax3.set_xlim(5.636, 5.704)
    ax3.set_ylim(0.2434, 0.2592)
    ax3.set_xticks([5.64, 5.66, 5.68, 5.70])
    ax3.set_yticks([0.244, 0.250, 0.256])
    clean_axes(ax3)

    savefig(fig, out_dir, "fig2_parameter_scheduler_sci", dpi)

def draw_fig3(human: pd.DataFrame, out_dir: Path, dpi: int):
    """
    Fig. 3: human evaluation.

    Compact two-panel layout:
    - left: four object-level error rates;
    - right: object correctness with narrower bars;
    - moderate spacing between panels;
    - legend placed inside the left panel to reduce blank space.
    """
    method_col = find_col(human, ["Method", "method"])
    hdf = human.set_index(method_col)

    error_metrics = [
        (find_col(human, ["Fusion Rate ↓ (%)"], contains=["fusion", "rate"]), "Fusion"),
        (find_col(human, ["Severe Fusion ↓ (%)"], contains=["severe", "fusion"]), "Severe Fusion"),
        (find_col(human, ["Missing Rate ↓ (%)"], contains=["missing"]), "Missing"),
        (find_col(human, ["Binding Error ↓ (%)"], contains=["binding"]), "Binding Error"),
    ]
    correct_metric = (find_col(human, ["Object Correct ↑ (%)"], contains=["object", "correct"]), "Object Correctness")

    ebama_err = np.array([float(hdf.loc["EBAMA", col]) for col, _ in error_metrics])
    sa_err = np.array([float(hdf.loc["SA-EBAMA", col]) for col, _ in error_metrics])
    ebama_correct = float(hdf.loc["EBAMA", correct_metric[0]])
    sa_correct = float(hdf.loc["SA-EBAMA", correct_metric[0]])

    fig = plt.figure(figsize=(8.45, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.75, 0.92], wspace=0.22)

    # Left panel: object-level error rates.
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(error_metrics))
    width = 0.28

    b1 = ax1.bar(x - width / 2, ebama_err, width, label="EBAMA",
                 color=PAIR_COLORS["EBAMA"], edgecolor="none", linewidth=0.0, alpha=0.88)
    b2 = ax1.bar(x + width / 2, sa_err, width, label="SA-EBAMA",
                 color=PAIR_COLORS["SA-EBAMA"], edgecolor="none", linewidth=0.0, alpha=0.88)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lab for _, lab in error_metrics], rotation=0, ha="center", fontsize=9)
    ax1.set_ylabel("Error Rate (%)", fontsize=10)
    ax1.set_ylim(0, 30)
    ax1.set_title("Object-Level Error Rates", pad=3, fontweight="bold", fontsize=10)
    ax1.tick_params(axis="y", labelsize=9)
    clean_axes(ax1)

    # Keep the legend inside the empty upper-right area of the left panel.
    ax1.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=1, handlelength=1.4, fontsize=9)

    for bars, is_sa in [(b1, False), (b2, True)]:
        for b in bars:
            v = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2,
                v + 0.75,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_sa else "normal",
            )

    # Right panel: object correctness.
    ax2 = fig.add_subplot(gs[0, 1])
    x2 = np.array([0.0])

    # Match the *visual pixel width* of the right-panel bars to the bars
    # in the left grouped bar chart. Because the right panel is narrower
    # than the left panel, using the same data-unit width would not yield
    # the same apparent bar width.
    right_xlim = (-0.85, 0.85)
    left_xspan = ax1.get_xlim()[1] - ax1.get_xlim()[0]
    right_xspan = right_xlim[1] - right_xlim[0]
    left_panel_width = ax1.get_position().width
    right_panel_width = ax2.get_position().width
    width2 = width * (right_xspan / left_xspan) * (left_panel_width / right_panel_width)

    b3 = ax2.bar(x2 - width2 / 2, [ebama_correct], width2, label="EBAMA",
                 color=PAIR_COLORS["EBAMA"], edgecolor="none", linewidth=0.0, alpha=0.88)
    b4 = ax2.bar(x2 + width2 / 2, [sa_correct], width2, label="SA-EBAMA",
                 color=PAIR_COLORS["SA-EBAMA"], edgecolor="none", linewidth=0.0, alpha=0.88)

    ax2.set_xlim(*right_xlim)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([correct_metric[1]], rotation=0, ha="center", fontsize=9)
    ax2.set_ylabel("Correctness Rate (%)", fontsize=10)
    ax2.set_ylim(90, 100)
    ax2.set_title("Object Correctness", pad=3, fontweight="bold", fontsize=10)
    ax2.tick_params(axis="y", labelsize=9)
    clean_axes(ax2)

    for bars, is_sa in [(b3, False), (b4, True)]:
        for b in bars:
            v = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2,
                v + 0.20,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_sa else "normal",
            )

    # Moderate margins; avoid extra lines/objects around the x-axis.
    fig.subplots_adjust(left=0.070, right=0.985, top=0.875, bottom=0.22, wspace=0.22)
    savefig(fig, out_dir, "fig3_human_evaluation_sci", dpi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table4", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--human", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/paper_figures/table_to_figures_sci")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_matplotlib()
    out_dir = Path(args.out_dir)

    draw_fig1(pd.read_csv(args.table4), out_dir, args.dpi)
    draw_fig2(pd.read_csv(args.mask), pd.read_csv(args.schedule), out_dir, args.dpi)
    draw_fig3(pd.read_csv(args.human), out_dir, args.dpi)


if __name__ == "__main__":
    main()
