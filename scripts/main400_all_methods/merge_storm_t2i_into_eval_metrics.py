#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

metric_dir = Path("results/formal_main400_all_methods/metrics")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=Path, default=metric_dir / "eval_all_metrics_with_storm_skip_t2i.csv")
    parser.add_argument("--t2i_csv", type=Path, default=metric_dir / "t2i_spatial_scores_per_image_with_storm.csv")
    parser.add_argument("--out_csv", type=Path, default=metric_dir / "eval_all_metrics_with_storm_with_t2i.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.eval_csv)
    t2i = pd.read_csv(args.t2i_csv)

    seed_order = [1, 1234, 42, 70690]
    seed_index = t2i["seed_index"].astype(int)
    if not seed_index.between(0, len(seed_order) - 1).all():
        raise ValueError("Spatial-score seed_index is outside the four-seed protocol.")
    t2i["seed"] = seed_index.map(dict(enumerate(seed_order)))
    if "raw_method" in t2i:
        t2i["method"] = t2i["raw_method"].astype(str)
    else:
        aliases = {"SD": "sd", "Attend-and-Excite": "attend_excite", "SynGen": "syngen",
                   "CONFORM": "conform", "STORM": "storm", "EBAMA": "ebama",
                   "EBAMA+Sep": "ebama_sep", "EBAMA+Adaptive": "ebama_adaptive", "SA-EBAMA": "sa_ebama"}
        t2i["method"] = t2i["method"].map(aliases).fillna(t2i["method"].astype(str).str.lower())
        if "source_image_path" in t2i:
            seed_from_name = t2i["source_image_path"].astype(str).str.extract(r"seed(\d+)\.png$", expand=False)
            t2i["seed"] = pd.to_numeric(seed_from_name, errors="coerce").fillna(t2i["seed"]).astype(int)

    keep = ["method", "prompt_id", "seed", "t2i_spatial"]
    t2i_keep = t2i[keep].copy()


    if "t2i_spatial" in df.columns:
        df = df.drop(columns=["t2i_spatial"])

    merged = df.merge(
        t2i_keep,
        on=["method", "prompt_id", "seed"],
        how="left",
        validate="one_to_one",
    )

    out = args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print("saved:", out)
    print("rows:", len(merged))
    print("T2I non-null by method:")
    print(merged.groupby("method")["t2i_spatial"].count().to_string())

    print("\nT2I mean by method:")
    print(merged.groupby("method")["t2i_spatial"].mean().to_string())


if __name__ == "__main__":
    main()
