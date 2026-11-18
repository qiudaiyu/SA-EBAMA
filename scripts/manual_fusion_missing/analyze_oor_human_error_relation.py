#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def norm_method(x):
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if x in ["saebama", "sa_ebama", "sa_ebama_v2"]:
        return "sa_ebama"
    if x == "ebama":
        return "ebama"
    return x


def to_binary_error(x):
    """
    Convert majority-vote resolved_score to binary visible error.
    0 -> no error
    1 or 2 -> error
    -1 or invalid -> NaN
    """
    if pd.isna(x):
        return np.nan
    try:
        v = int(float(x))
    except Exception:
        return np.nan

    if v == -1:
        return np.nan
    if v == 0:
        return 0
    if v in [1, 2]:
        return 1
    return np.nan


def delta_to_group(d):
    if pd.isna(d):
        return "invalid"
    if d > 0:
        return "improved"
    if d < 0:
        return "worsened"
    return "unchanged"


def bootstrap_ci(values, n_boot=5000, seed=123):
    values = pd.Series(values).dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def safe_rate(mask):
    if len(mask) == 0:
        return np.nan
    return float(np.mean(mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True,
                        help="eval_metrics_final.csv containing method, prompt_id, seed, oor.")
    parser.add_argument("--agreement_detail_csv", required=True,
                        help="manual_fusion_missing_agreement_detail.csv with resolved_score.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--make_figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # 1. Load OOR from formal metrics

    metrics = pd.read_csv(args.metrics_csv)

    required_metrics_cols = ["method", "dataset", "prompt_id", "prompt", "seed", "oor"]
    for c in required_metrics_cols:
        if c not in metrics.columns:
            raise ValueError(f"Missing column in metrics_csv: {c}")

    metrics["method"] = metrics["method"].apply(norm_method)
    metrics["seed"] = metrics["seed"].astype(str)

    oor = metrics[metrics["method"].isin(["ebama", "sa_ebama"])].copy()
    oor = oor.dropna(subset=["oor"])

    oor_wide = (
        oor.pivot_table(
            index=["dataset", "prompt_id", "seed", "prompt"],
            columns="method",
            values="oor",
            aggfunc="mean"
        )
        .reset_index()
        .rename(columns={"ebama": "oor_ebama", "sa_ebama": "oor_sa_ebama"})
    )

    if "oor_ebama" not in oor_wide.columns or "oor_sa_ebama" not in oor_wide.columns:
        raise ValueError("Cannot find both EBAMA and SA-EBAMA OOR after pivoting.")

    oor_wide["delta_oor"] = oor_wide["oor_ebama"] - oor_wide["oor_sa_ebama"]
    oor_wide.to_csv(out_dir / "oor_pairs_ebama_sa.csv", index=False, encoding="utf-8-sig")


    # 2. Load human majority-vote labels

    detail = pd.read_csv(args.agreement_detail_csv)

    required_detail_cols = ["pair_id", "dataset", "method", "label", "resolved_score"]
    for c in required_detail_cols:
        if c not in detail.columns:
            raise ValueError(f"Missing column in agreement_detail_csv: {c}")

    detail["method"] = detail["method"].apply(norm_method)
    detail["label"] = detail["label"].astype(str).str.strip().str.lower()
    detail["pair_id"] = detail["pair_id"].astype(str)

    # Extract prompt_id and seed from pair_id, e.g., abc_0001__seed1
    detail["prompt_id"] = detail["pair_id"].str.replace(r"__seed\d+$", "", regex=True)
    detail["seed"] = detail["pair_id"].str.extract(r"__seed(\d+)")[0].astype(str)

    # Convert long format to one row per pair-method
    human_long = (
        detail.pivot_table(
            index=["pair_id", "dataset", "prompt_id", "seed", "method"],
            columns="label",
            values="resolved_score",
            aggfunc="first"
        )
        .reset_index()
    )

    required_labels = ["fusion", "missing", "binding_error", "object_correctness"]
    for lab in required_labels:
        if lab not in human_long.columns:
            raise ValueError(f"Missing label in human annotation: {lab}")

    # Convert to one row per pair, columns = label_method
    human_pair = (
        human_long.pivot_table(
            index=["pair_id", "dataset", "prompt_id", "seed"],
            columns="method",
            values=required_labels,
            aggfunc="first"
        )
    )

    human_pair.columns = [f"{label}_{method}" for label, method in human_pair.columns]
    human_pair = human_pair.reset_index()

    # Make sure required pair columns exist
    required_pair_cols = [
        "fusion_ebama", "fusion_sa_ebama",
        "missing_ebama", "missing_sa_ebama",
        "binding_error_ebama", "binding_error_sa_ebama",
        "object_correctness_ebama", "object_correctness_sa_ebama"
    ]
    for c in required_pair_cols:
        if c not in human_pair.columns:
            raise ValueError(f"Missing human pair column after pivot: {c}")

    human_pair.to_csv(out_dir / "human_pair_majority_labels.csv", index=False, encoding="utf-8-sig")


    # 3. Merge OOR and human labels

    merged = human_pair.merge(
        oor_wide[["prompt_id", "seed", "prompt", "oor_ebama", "oor_sa_ebama", "delta_oor"]],
        on=["prompt_id", "seed"],
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError("Merged result is empty. Please check prompt_id and seed formats.")


    # 4. Compute human error changes

    for err in ["fusion", "missing", "binding_error"]:
        merged[f"{err}_ebama_bin"] = merged[f"{err}_ebama"].apply(to_binary_error)
        merged[f"{err}_sa_ebama_bin"] = merged[f"{err}_sa_ebama"].apply(to_binary_error)

        merged[f"delta_{err}"] = (
            merged[f"{err}_ebama_bin"] - merged[f"{err}_sa_ebama_bin"]
        )

        merged[f"{err}_change"] = merged[f"delta_{err}"].apply(delta_to_group)

    # optional severe fusion
    merged["severe_fusion_ebama_bin"] = merged["fusion_ebama"].apply(
        lambda x: 1 if not pd.isna(x) and int(float(x)) == 2 else 0
    )
    merged["severe_fusion_sa_ebama_bin"] = merged["fusion_sa_ebama"].apply(
        lambda x: 1 if not pd.isna(x) and int(float(x)) == 2 else 0
    )
    merged["delta_severe_fusion"] = (
        merged["severe_fusion_ebama_bin"] - merged["severe_fusion_sa_ebama_bin"]
    )
    merged["severe_fusion_change"] = merged["delta_severe_fusion"].apply(delta_to_group)

    merged.to_csv(out_dir / "oor_human_pair_level.csv", index=False, encoding="utf-8-sig")


    # 5. Summary: ΔOOR grouped by human error change

    rows = []
    for err in ["binding_error", "fusion", "missing", "severe_fusion"]:
        for group in ["improved", "unchanged", "worsened"]:
            sub = merged[merged[f"{err}_change"] == group].copy()
            ci_low, ci_high = bootstrap_ci(sub["delta_oor"])
            rows.append({
                "Human error type": err,
                "Error change": group,
                "n": len(sub),
                "Mean ΔOOR ↑": sub["delta_oor"].mean(),
                "Median ΔOOR ↑": sub["delta_oor"].median(),
                "95% CI Low": ci_low,
                "95% CI High": ci_high,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "oor_human_error_change_summary.csv", index=False, encoding="utf-8-sig")

    # Paper-ready simplified table: Binding, Fusion, Missing only
    paper = summary[summary["Human error type"].isin(["binding_error", "fusion", "missing"])].copy()
    paper["Human error type"] = paper["Human error type"].replace({
        "binding_error": "Binding Error",
        "fusion": "Fusion",
        "missing": "Missing"
    })
    paper.to_csv(out_dir / "paper_oor_human_error_relation_table.csv", index=False, encoding="utf-8-sig")


    # 6. OOR reduction tertile analysis

    valid = merged.dropna(subset=["delta_oor"]).copy()
    valid["OOR reduction group"] = pd.qcut(
        valid["delta_oor"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    )

    tertile_rows = []
    for group, sub in valid.groupby("OOR reduction group", observed=True):
        row = {
            "OOR reduction group": group,
            "n": len(sub),
            "Mean ΔOOR ↑": sub["delta_oor"].mean(),
        }
        for err in ["binding_error", "fusion", "missing"]:
            row[f"{err}_improved_rate_%"] = 100 * safe_rate(sub[f"{err}_change"] == "improved")
            row[f"{err}_worsened_rate_%"] = 100 * safe_rate(sub[f"{err}_change"] == "worsened")
        tertile_rows.append(row)

    tertile = pd.DataFrame(tertile_rows)
    tertile.to_csv(out_dir / "oor_reduction_tertile_error_rates.csv", index=False, encoding="utf-8-sig")


    # 7. Spearman correlation, exploratory only

    corr_rows = []
    for err in ["binding_error", "fusion", "missing"]:
        tmp = merged[["delta_oor", f"delta_{err}"]].dropna()
        rho = tmp["delta_oor"].corr(tmp[f"delta_{err}"], method="spearman") if len(tmp) > 2 else np.nan
        corr_rows.append({
            "Human error type": err,
            "n": len(tmp),
            "Spearman rho": rho
        })

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "oor_human_error_spearman.csv", index=False, encoding="utf-8-sig")


    # 8. Overall diagnostic summary

    overall = pd.DataFrame([{
        "n_merged_pairs": len(merged),
        "mean_delta_oor": merged["delta_oor"].mean(),
        "median_delta_oor": merged["delta_oor"].median(),
        "delta_oor_positive_n": int((merged["delta_oor"] > 0).sum()),
        "delta_oor_positive_rate_%": 100 * float((merged["delta_oor"] > 0).mean()),
        "delta_oor_zero_n": int((merged["delta_oor"] == 0).sum()),
        "delta_oor_negative_n": int((merged["delta_oor"] < 0).sum()),
    }])
    overall.to_csv(out_dir / "oor_human_overall_summary.csv", index=False, encoding="utf-8-sig")


    # 9. Figures

    if args.make_figures:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)

        for ax, err, title in [
            (axes[0], "binding_error", "Binding Error"),
            (axes[1], "fusion", "Fusion")
        ]:
            data = [
                merged[merged[f"{err}_change"] == g]["delta_oor"].dropna().values
                for g in ["improved", "unchanged", "worsened"]
            ]
            ax.boxplot(data, labels=["Improved", "Unchanged", "Worsened"], showmeans=True)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Human error change")
            ax.tick_params(axis="x", rotation=20)

        axes[0].set_ylabel("ΔOOR = OOR_EBAMA − OOR_SA-EBAMA")
        plt.tight_layout()
        plt.savefig(out_dir / "oor_vs_human_error_boxplot.pdf", bbox_inches="tight")
        plt.savefig(out_dir / "oor_vs_human_error_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Bar figure for mean ΔOOR
        fig_df = paper.copy()
        fig_df = fig_df[fig_df["Error change"].isin(["improved", "unchanged", "worsened"])]

        plt.figure(figsize=(7.2, 3.6))
        x_labels = []
        y_vals = []
        for err in ["Binding Error", "Fusion", "Missing"]:
            for group in ["improved", "unchanged", "worsened"]:
                sub = fig_df[(fig_df["Human error type"] == err) & (fig_df["Error change"] == group)]
                if len(sub) > 0:
                    x_labels.append(f"{err}\n{group}")
                    y_vals.append(float(sub["Mean ΔOOR ↑"].iloc[0]) if not pd.isna(sub["Mean ΔOOR ↑"].iloc[0]) else 0.0)

        bars = plt.bar(range(len(x_labels)), y_vals)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
        plt.ylabel("Mean ΔOOR")
        plt.title("OOR Reduction Grouped by Human Error Changes")
        for b, v in zip(bars, y_vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "mean_delta_oor_by_human_error_change.pdf", bbox_inches="tight")
        plt.savefig(out_dir / "mean_delta_oor_by_human_error_change.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nDone.")
    print("Outputs saved to:", out_dir)
    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nOOR-human relation summary:")
    print(paper.to_string(index=False))
    print("\nTertile summary:")
    print(tertile.to_string(index=False))
    print("\nSpearman:")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
