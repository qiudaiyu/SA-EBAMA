#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_per_dataset", type=int, default=25)
    parser.add_argument("--methods", nargs="+", default=["ebama", "sa_ebama"])
    parser.add_argument("--seed_priority", nargs="+", type=int, default=[1, 42, 1234, 70690])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--require_exists", type=int, default=1)
    return parser.parse_args()


def normalize_exists(x):
    try:
        return int(x) == 1
    except Exception:
        return str(x).lower() in {"true", "yes", "1"}


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)

    required_cols = [
        "method", "dataset", "prompt_id", "prompt",
        "seed", "image_path", "exists"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"index_csv missing required column: {c}")

    df = df[df["method"].isin(args.methods)].copy()
    df = df[df["exists"].apply(normalize_exists)].copy()
    df["seed"] = df["seed"].astype(int)

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    print("[INFO] datasets:", datasets)
    print("[INFO] methods:", sorted(df["method"].unique().tolist()))

    selected_pairs = []

    for dataset in datasets:
        sub = df[df["dataset"] == dataset].copy()
        prompt_ids = sorted(sub["prompt_id"].dropna().unique().tolist())
        rng.shuffle(prompt_ids)

        dataset_pairs = []

        for prompt_id in prompt_ids:
            ok_pair = None

            for seed in args.seed_priority:
                rows = sub[(sub["prompt_id"] == prompt_id) & (sub["seed"] == seed)]
                method_set = set(rows["method"].tolist())

                if all(m in method_set for m in args.methods):
                    pair_rows = {}
                    valid = True

                    for method in args.methods:
                        r = rows[rows["method"] == method].iloc[0]
                        image_path = str(r["image_path"])
                        if not Path(image_path).exists():
                            valid = False
                            break
                        pair_rows[method] = r

                    if valid:
                        ok_pair = (seed, pair_rows)
                        break

            if ok_pair is None:
                continue

            seed, pair_rows = ok_pair
            r0 = list(pair_rows.values())[0]

            left_method, right_method = args.methods
            if rng.random() < 0.5:
                left_method, right_method = right_method, left_method

            pair_id = f"{prompt_id}__seed{seed}"

            dataset_pairs.append({
                "pair_id": pair_id,
                "dataset": dataset,
                "prompt_id": prompt_id,
                "prompt": r0["prompt"],
                "seed": seed,
                "left_method": left_method,
                "left_image_path": pair_rows[left_method]["image_path"],
                "right_method": right_method,
                "right_image_path": pair_rows[right_method]["image_path"],
            })

            if len(dataset_pairs) >= args.n_per_dataset:
                break

        if len(dataset_pairs) < args.n_per_dataset:
            print(f"[WARN] dataset={dataset}: only found {len(dataset_pairs)} complete pairs")

        selected_pairs.extend(dataset_pairs[:args.n_per_dataset])

    pairs = pd.DataFrame(selected_pairs)
    if len(pairs) == 0:
        raise RuntimeError("No valid pairs found. Please check index_csv, methods, seeds, image_path, exists.")

    pairs = pairs.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
    pairs["task_order"] = range(1, len(pairs) + 1)

    pairs_csv = out_dir / "manual_fusion_missing_pairs_100.csv"
    pairs.to_csv(pairs_csv, index=False, encoding="utf-8")

    image_rows = []
    for _, r in pairs.iterrows():
        for side in ["left", "right"]:
            method = r[f"{side}_method"]
            image_path = r[f"{side}_image_path"]
            image_rows.append({
                "pair_id": r["pair_id"],
                "dataset": r["dataset"],
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "seed": r["seed"],
                "method": method,
                "image_path": image_path,
                "display_side": side,
            })

    tasks = pd.DataFrame(image_rows)
    tasks_csv = out_dir / "manual_fusion_missing_tasks_100.csv"
    tasks.to_csv(tasks_csv, index=False, encoding="utf-8")

    print("[OK] saved:", pairs_csv)
    print("[OK] saved:", tasks_csv)
    print("[INFO] total pairs:", len(pairs))
    print("[INFO] dataset counts:")
    print(pairs["dataset"].value_counts())


if __name__ == "__main__":
    main()
