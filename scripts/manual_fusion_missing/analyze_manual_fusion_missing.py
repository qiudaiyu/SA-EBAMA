#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_ci(values, n_boot=1000, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def normalize_method(x):
    x = str(x).strip()
    low = x.lower()
    if low in {"ebama", "e-bama"}:
        return "EBAMA"
    if low in {"sa_ebama", "sa-ebama", "saebama"}:
        return "SA-EBAMA"
    return x


def main():
    args = parse_args()

    resolved_csv = Path(args.resolved_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolved_csv)
    df["method"] = df["method"].apply(normalize_method)

    required = [
        "pair_id", "dataset", "method",
        "fusion_score", "missing_score",
        "binding_error_score", "object_correctness_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"resolved csv missing columns: {missing}")


    df["fusion_problem"] = (df["fusion_score"] >= 1).astype(int)
    df["severe_fusion"] = (df["fusion_score"] == 2).astype(int)
    df["missing_problem"] = (df["missing_score"] >= 1).astype(int)
    df["severe_missing"] = (df["missing_score"] == 2).astype(int)
    df["binding_problem"] = (df["binding_error_score"] >= 1).astype(int)
    df["object_correct"] = (df["object_correctness_score"] == 1).astype(int)


    method_summary = (
        df.groupby("method", as_index=False)
        .agg(
            n=("pair_id", "nunique"),
            fusion_rate=("fusion_problem", "mean"),
            severe_fusion_rate=("severe_fusion", "mean"),
            missing_rate=("missing_problem", "mean"),
            severe_missing_rate=("severe_missing", "mean"),
            binding_error_rate=("binding_problem", "mean"),
            object_correctness_rate=("object_correct", "mean"),
            fusion_score_mean=("fusion_score", "mean"),
            missing_score_mean=("missing_score", "mean"),
            binding_error_score_mean=("binding_error_score", "mean"),
        )
    )

    method_summary.to_csv(out_dir / "manual_fusion_missing_method_summary.csv", index=False, encoding="utf-8")

    # 2. by-dataset summary
    by_dataset = (
        df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("pair_id", "nunique"),
            fusion_rate=("fusion_problem", "mean"),
            severe_fusion_rate=("severe_fusion", "mean"),
            missing_rate=("missing_problem", "mean"),
            severe_missing_rate=("severe_missing", "mean"),
            binding_error_rate=("binding_problem", "mean"),
            object_correctness_rate=("object_correct", "mean"),
        )
    )

    by_dataset.to_csv(out_dir / "manual_fusion_missing_by_dataset.csv", index=False, encoding="utf-8")

    # 3. paired summary

    pair_metrics = {
        "Fusion": "fusion_problem",
        "Missing": "missing_problem",
        "Binding Error": "binding_problem",
        "Object Correctness": "object_correct",
    }

    paired_rows = []

    for metric_name, col in pair_metrics.items():
        pivot = df.pivot_table(
            index=["pair_id", "dataset"],
            columns="method",
            values=col,
            aggfunc="first"
        ).reset_index()

        if "EBAMA" not in pivot.columns or "SA-EBAMA" not in pivot.columns:
            print(f"[WARN] skip {metric_name}: EBAMA or SA-EBAMA missing")
            continue

        valid = pivot.dropna(subset=["EBAMA", "SA-EBAMA"]).copy()

        if metric_name == "Object Correctness":

            delta = valid["SA-EBAMA"].astype(float) - valid["EBAMA"].astype(float)
            win = (valid["SA-EBAMA"] > valid["EBAMA"]).astype(int)
            tie = (valid["SA-EBAMA"] == valid["EBAMA"]).astype(int)
        else:

            delta = valid["EBAMA"].astype(float) - valid["SA-EBAMA"].astype(float)
            win = (valid["EBAMA"] > valid["SA-EBAMA"]).astype(int)
            tie = (valid["EBAMA"] == valid["SA-EBAMA"]).astype(int)

        ci_low, ci_high = bootstrap_ci(delta.values, n_boot=args.n_boot, seed=args.seed)

        paired_rows.append({
            "metric": metric_name,
            "mean_delta": float(delta.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "win_rate": float(win.mean()),
            "tie_rate": float(tie.mean()),
            "n": int(len(valid)),
        })

    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(out_dir / "manual_fusion_missing_paired_summary.csv", index=False, encoding="utf-8")

    print("[OK] saved:", out_dir / "manual_fusion_missing_method_summary.csv")
    print("[OK] saved:", out_dir / "manual_fusion_missing_by_dataset.csv")
    print("[OK] saved:", out_dir / "manual_fusion_missing_paired_summary.csv")


if __name__ == "__main__":
    main()
