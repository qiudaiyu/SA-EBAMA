#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def norm_method(x):
    x = str(x).strip()
    x = x.replace("_", "-")
    mapping = {
        "sd": "SD",
        "stable-diffusion": "SD",
        "attend-excite": "Attend-and-Excite",
        "attend-and-excite": "Attend-and-Excite",
        "syngen": "SynGen",
        "conform": "CONFORM",
        "storm": "STORM",
        "ebama": "EBAMA",
        "ebama-sep": "EBAMA+Sep",
        "ebama-adaptive": "EBAMA+Adaptive",
        "sa-ebama": "SA-EBAMA",
    }
    return mapping.get(x.lower(), x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.runtime_csv)
    if "exists" in df.columns:
        df = df.loc[df["exists"].eq(1)].copy()
    if df.empty:
        raise ValueError("The runtime CSV contains no generated samples.")

    method_col = find_col(df, ["method"])
    time_col = find_col(df, ["time_seconds", "time_per_image", "time/image", "time_image", "elapsed_time", "runtime", "seconds"])
    update_col = find_col(df, ["num_updated_steps", "update_steps", "updates", "latent_update_steps", "n_updates"])

    if method_col is None:
        raise ValueError("Cannot find method column.")
    if time_col is None:
        raise ValueError("Cannot find runtime/time column.")
    if update_col is None:
        print("Warning: update_steps column not found. Only time will be summarized.")

    df["method_norm"] = df[method_col].apply(norm_method)
    df["time_sec"] = pd.to_numeric(df[time_col], errors="coerce")

    if update_col:
        df["update_steps_num"] = pd.to_numeric(df[update_col], errors="coerce")
    else:
        df["update_steps_num"] = pd.NA

    rows = []
    method_order = [
        "SD",
        "Attend-and-Excite",
        "SynGen",
        "CONFORM",
        "STORM",
        "EBAMA",
        "EBAMA+Sep",
        "EBAMA+Adaptive",
        "SA-EBAMA",
    ]

    for method in method_order:
        sub = df[df["method_norm"] == method].copy()
        if len(sub) == 0:
            continue

        row = {
            "Method": method,
            "n": len(sub),
            "Time/Image mean (s)": sub["time_sec"].mean(),
            "Time/Image std (s)": sub["time_sec"].std(),
            "Time/Image median (s)": sub["time_sec"].median(),
        }

        if update_col:
            row["Update Steps mean"] = sub["update_steps_num"].mean()
            row["Update Steps std"] = sub["update_steps_num"].std()
            row["Update Steps median"] = sub["update_steps_num"].median()

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "runtime_summary_by_method.csv", index=False, encoding="utf-8-sig")

    # Paper-ready table: only efficiency-related methods
    paper_methods = ["EBAMA", "EBAMA+Sep", "EBAMA+Adaptive", "SA-EBAMA"]
    paper = summary[summary["Method"].isin(paper_methods)].copy()
    paper.to_csv(out_dir / "paper_runtime_protocol_table.csv", index=False, encoding="utf-8-sig")

    print("\nRuntime summary saved to:", out_dir / "runtime_summary_by_method.csv")
    print(summary.to_string(index=False))

    print("\nPaper runtime table saved to:", out_dir / "paper_runtime_protocol_table.csv")
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
