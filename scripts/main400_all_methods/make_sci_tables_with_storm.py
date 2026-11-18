#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "sd",
    "attend_excite",
    "syngen",
    "conform",
    "storm",
    "ebama",
    "ebama_sep",
    "ebama_adaptive",
    "sa_ebama",
]

INTERNAL_ORDER = [
    "ebama",
    "ebama_sep",
    "ebama_adaptive",
    "sa_ebama",
]


def bootstrap_ci(x, n_boot=1000, seed=42):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(rng.choice(x, size=len(x), replace=True).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_table(df):
    df = df.copy()
    for c in df.columns:
        if c in ["method", "dataset", "Sep", "Adaptive", "metric"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(6)
    return df


def order_methods(df, order):
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    return df.sort_values("method")


def make_main_table(df):
    agg = {
        "n": ("image_path", "count"),
        "CLIP Full↑": ("clip_full", "mean"),
        "CLIP Min↑": ("clip_min", "mean"),
        "CLIP Mean↑": ("clip_mean", "mean"),
        "T-C Sim↑": ("tc_sim", "mean"),
        "Weak-object Risk Proxy↓": ("weak_object_risk_proxy", "mean"),
        "Time/Image↓": ("time_seconds", "mean"),
    }

    out = df.groupby("method").agg(**agg).reset_index()
    out["T2I-Spatial↑"] = "TBD"


    cols = [
        "method", "n",
        "CLIP Full↑", "CLIP Min↑", "CLIP Mean↑",
        "T-C Sim↑", "T2I-Spatial↑",
        "Weak-object Risk Proxy↓", "Time/Image↓"
    ]
    out = out[cols]

    return fmt_table(order_methods(out, METHOD_ORDER))


def make_by_dataset_table(df):
    rows = []

    for (dataset, method), g in df.groupby(["dataset", "method"]):
        row = {
            "dataset": dataset,
            "method": method,
            "n": len(g),
            "CLIP Min↑": pd.to_numeric(g["clip_min"], errors="coerce").mean(),
            "CLIP Mean↑": pd.to_numeric(g["clip_mean"], errors="coerce").mean(),
            "T-C Sim↑": pd.to_numeric(g["tc_sim"], errors="coerce").mean() if "tc_sim" in g.columns else np.nan,
            "Weak-object Risk Proxy↓": pd.to_numeric(g["weak_object_risk_proxy"], errors="coerce").mean(),
            "T2I-Spatial↑": "TBD",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values(["dataset", "method"])
    return fmt_table(out)


def make_internal_ablation_table(df):
    sub = df[df["method"].isin(INTERNAL_ORDER)].copy()

    agg = {
        "n": ("image_path", "count"),
        "CLIP Min↑": ("clip_min", "mean"),
        "T-C Sim↑": ("tc_sim", "mean"),
        "OOR↓": ("oor", "mean"),
        "Weak-object Risk Proxy↓": ("weak_object_risk_proxy", "mean"),
        "Update Steps↓": ("num_updated_steps", "mean"),
        "Time/Image↓": ("time_seconds", "mean"),
        "Peak Memory↓": ("peak_memory_mb", "mean"),
    }

    out = sub.groupby("method").agg(**agg).reset_index()

    out.insert(1, "Sep", out["method"].map({
        "ebama": "No",
        "ebama_sep": "Yes",
        "ebama_adaptive": "No",
        "sa_ebama": "Yes",
    }))

    out.insert(2, "Adaptive", out["method"].map({
        "ebama": "No",
        "ebama_sep": "No",
        "ebama_adaptive": "Yes",
        "sa_ebama": "Yes",
    }))

    out["T2I-Spatial↑"] = "TBD"

    cols = [
        "method", "Sep", "Adaptive", "n",
        "CLIP Min↑", "T-C Sim↑", "OOR↓",
        "Weak-object Risk Proxy↓",
        "Update Steps↓", "Time/Image↓",
        "Peak Memory↓", "T2I-Spatial↑"
    ]
    out = out[cols]

    return fmt_table(order_methods(out, INTERNAL_ORDER))


def make_paired_table(df, internal_only=False):
    keys = ["prompt_id", "seed"]
    eb = df[df["method"] == "ebama"].copy()

    if internal_only:
        methods = ["ebama_sep", "ebama_adaptive", "sa_ebama"]
    else:
        methods = ["sd", "attend_excite", "syngen", "conform", "storm", "ebama_sep", "ebama_adaptive", "sa_ebama"]

    metrics = {
        "clip_min": ("ΔCLIP Min↑", "high"),
        "clip_full": ("ΔCLIP Full↑", "high"),
        "clip_mean": ("ΔCLIP Mean↑", "high"),
        "tc_sim": ("ΔT-C Sim↑", "high"),
        "weak_object_risk_proxy": ("ΔRisk Proxy↑", "low"),
        "time_seconds": ("ΔTime↑", "low"),
    }

    if internal_only:
        metrics["oor"] = ("ΔOOR↑", "low")
        metrics["num_updated_steps"] = ("ΔSteps↑", "low")

    rows = []

    for method in methods:
        mg = df[df["method"] == method].copy()
        pair = mg.merge(eb, on=keys, suffixes=("_method", "_ebama"), how="inner")

        for col, (name, direction) in metrics.items():
            cm = f"{col}_method"
            ce = f"{col}_ebama"

            if cm not in pair.columns or ce not in pair.columns:
                continue

            a = pd.to_numeric(pair[cm], errors="coerce")
            b = pd.to_numeric(pair[ce], errors="coerce")
            valid = a.notna() & b.notna()

            if valid.sum() == 0:
                continue

            if direction == "high":
                delta = a[valid] - b[valid]
            else:
                delta = b[valid] - a[valid]

            lo, hi = bootstrap_ci(delta)

            rows.append({
                "method": method,
                "metric": name,
                "n": int(valid.sum()),
                "mean_delta": float(delta.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "win_rate": float((delta > 0).mean()),
            })


    if not internal_only:
        for method in methods:
            rows.append({
                "method": method,
                "metric": "ΔT2I↑",
                "n": "TBD",
                "mean_delta": "TBD",
                "ci_low": "TBD",
                "ci_high": "TBD",
                "win_rate": "TBD",
            })
    else:
        for method in methods:
            rows.append({
                "method": method,
                "metric": "ΔT2I↑",
                "n": "TBD",
                "mean_delta": "TBD",
                "ci_low": "TBD",
                "ci_high": "TBD",
                "win_rate": "TBD",
            })

    out = pd.DataFrame(rows)
    return fmt_table(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)

    if "weak_object_risk_proxy" not in df.columns:
        if "missing_proxy" in df.columns:
            df["weak_object_risk_proxy"] = df["missing_proxy"]
        else:
            df["weak_object_risk_proxy"] = df["clip_mean"] - df["clip_min"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_main_table(df).to_csv(out_dir / "table1_main_comparison_sci_skip_t2i.csv", index=False)
    make_by_dataset_table(df).to_csv(out_dir / "table2_by_dataset_sci_skip_t2i.csv", index=False)
    make_paired_table(df, internal_only=False).to_csv(out_dir / "table3_paired_vs_ebama_sci_skip_t2i.csv", index=False)
    make_internal_ablation_table(df).to_csv(out_dir / "table4_internal_ablation_sci_skip_t2i.csv", index=False)
    make_paired_table(df, internal_only=True).to_csv(out_dir / "table5_internal_paired_sci_skip_t2i.csv", index=False)

    print("Saved tables to:", out_dir)
    for p in sorted(out_dir.glob("*skip_t2i.csv")):
        print(p)


if __name__ == "__main__":
    main()
