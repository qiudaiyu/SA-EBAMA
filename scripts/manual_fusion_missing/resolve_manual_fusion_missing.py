#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
from collections import Counter

import pandas as pd


NUMERIC_LABELS = [
    "fusion",
    "missing",
    "binding_error",
    "object_correctness",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--strategy", default="majority", choices=["majority"])
    parser.add_argument("--save_agreement", required=True)
    return parser.parse_args()


def majority_vote(values):
    """Resolve labels by majority vote, with a median fallback when votes differ."""


    vals = [int(v) for v in values if pd.notna(v)]
    if len(vals) == 0:
        return -1

    cnt = Counter(vals)
    top_val, top_n = cnt.most_common(1)[0]
    if top_n >= 2:
        return int(top_val)


    vals_sorted = sorted(vals)
    return int(vals_sorted[len(vals_sorted) // 2])


def exact_agreement(values):
    vals = [int(v) for v in values if pd.notna(v)]
    if len(vals) <= 1:
        return 1.0
    return 1.0 if len(set(vals)) == 1 else 0.0


def stack_annotations(annotation_dir: Path):
    files = sorted(annotation_dir.glob("annotator_*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotator_*.csv found in {annotation_dir}")

    rows = []

    for f in files:
        df = pd.read_csv(f)

        required = [
            "annotator_id", "pair_id", "dataset", "prompt_id", "prompt", "seed",
            "left_method", "left_image_path",
            "right_method", "right_image_path",
            "left_fusion", "left_missing", "left_binding_error", "left_object_correctness",
            "right_fusion", "right_missing", "right_binding_error", "right_object_correctness",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")

        for _, r in df.iterrows():
            base = {
                "annotator_id": r["annotator_id"],
                "pair_id": r["pair_id"],
                "dataset": r["dataset"],
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "seed": r["seed"],
            }

            rows.append({
                **base,
                "side": "left",
                "method": r["left_method"],
                "image_path": r["left_image_path"],
                "fusion": int(r["left_fusion"]),
                "missing": int(r["left_missing"]),
                "binding_error": int(r["left_binding_error"]),
                "object_correctness": int(r["left_object_correctness"]),
            })

            rows.append({
                **base,
                "side": "right",
                "method": r["right_method"],
                "image_path": r["right_image_path"],
                "fusion": int(r["right_fusion"]),
                "missing": int(r["right_missing"]),
                "binding_error": int(r["right_binding_error"]),
                "object_correctness": int(r["right_object_correctness"]),
            })

    long_df = pd.DataFrame(rows)
    return long_df


def main():
    args = parse_args()

    annotation_dir = Path(args.annotation_dir)
    out_csv = Path(args.out_csv)
    agreement_csv = Path(args.save_agreement)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agreement_csv.parent.mkdir(parents=True, exist_ok=True)

    long_df = stack_annotations(annotation_dir)

    resolved_rows = []
    agreement_rows = []

    group_cols = ["pair_id", "method"]

    for (pair_id, method), g in long_df.groupby(group_cols):
        first = g.iloc[0]

        row = {
            "pair_id": pair_id,
            "dataset": first["dataset"],
            "prompt_id": first["prompt_id"],
            "prompt": first["prompt"],
            "seed": first["seed"],
            "method": method,
            "image_path": first["image_path"],
            "n_annotators": g["annotator_id"].nunique(),
        }

        for label in NUMERIC_LABELS:
            values = g[label].tolist()
            row[f"{label}_score"] = majority_vote(values)
            row[f"{label}_agreement"] = exact_agreement(values)

            agreement_rows.append({
                "pair_id": pair_id,
                "dataset": first["dataset"],
                "method": method,
                "label": label,
                "n_annotators": g["annotator_id"].nunique(),
                "values": "/".join(map(str, [int(v) for v in values])),
                "exact_agreement": exact_agreement(values),
                "resolved_score": row[f"{label}_score"],
            })

        resolved_rows.append(row)

    resolved = pd.DataFrame(resolved_rows)
    agreement_detail = pd.DataFrame(agreement_rows)

    resolved.to_csv(out_csv, index=False, encoding="utf-8")


    agreement_summary = (
        agreement_detail
        .groupby("label", as_index=False)
        .agg(
            n=("exact_agreement", "size"),
            agreement_rate=("exact_agreement", "mean"),
        )
    )


    detail_csv = agreement_csv.with_name(agreement_csv.stem + "_detail.csv")
    agreement_summary.to_csv(agreement_csv, index=False, encoding="utf-8")
    agreement_detail.to_csv(detail_csv, index=False, encoding="utf-8")

    print("[OK] saved resolved:", out_csv)
    print("[OK] saved agreement summary:", agreement_csv)
    print("[OK] saved agreement detail:", detail_csv)
    print("[INFO] resolved rows:", len(resolved))
    print("[INFO] expected rows: 100 pairs × 2 methods = 200")


if __name__ == "__main__":
    main()
