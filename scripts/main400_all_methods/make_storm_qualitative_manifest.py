#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

eval_csv = "results/formal_main400_all_methods/metrics/eval_all_metrics_with_storm_skip_t2i.csv"
df = pd.read_csv(eval_csv)

methods = ["sd", "attend_excite", "syngen", "conform", "storm", "ebama", "sa_ebama"]

sub = df[df["method"].isin(methods) & (df["exists"] == 1)].copy()

count = sub.groupby(
    ["dataset", "prompt_id", "prompt", "seed"]
)["method"].nunique().reset_index()

full = count[count["method"] == len(methods)].copy()

selected = []
for dataset, g in full.groupby("dataset"):
    selected.append(g.head(8))

selected = pd.concat(selected, ignore_index=True)

rows = []
for _, r in selected.iterrows():
    g = sub[
        (sub["prompt_id"] == r["prompt_id"]) &
        (sub["seed"] == r["seed"]) &
        (sub["method"].isin(methods))
    ].copy()
    g["qual_group"] = f'{r["dataset"]}__{r["prompt_id"]}__seed{r["seed"]}'
    rows.append(g)

manifest = pd.concat(rows, ignore_index=True)

out = Path("results/formal_main400_all_methods/qualitative/storm_qualitative_manifest_32groups.csv")
out.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(out, index=False)

print("saved:", out)
print("groups:", manifest["qual_group"].nunique())
print("rows:", len(manifest))
