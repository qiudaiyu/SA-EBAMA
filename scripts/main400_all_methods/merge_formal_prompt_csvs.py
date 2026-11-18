#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

import pandas as pd


REQUIRED = [
    "dataset", "split", "prompt_id", "prompt", "subprompts_json",
    "object_count", "attribute_count", "source", "subtype",
    "has_color", "has_spatial", "difficulty_tag",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--out_csv", type=str, default="data/formal_prompts/formal_prompts.csv")
    args = parser.parse_args()

    dfs = []
    for p in args.csvs:
        df = pd.read_csv(p)
        for col in REQUIRED:
            if col not in df.columns:
                raise ValueError(f"{p} missing required column: {col}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)


    if merged["prompt_id"].duplicated().any():
        raise ValueError("Duplicated prompt_id found. Please check input CSVs.")


    merged = merged.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8")

    print(f"Saved merged formal prompts to {out}")
    print("Total:", len(merged))
    print(merged["dataset"].value_counts())


if __name__ == "__main__":
    main()
