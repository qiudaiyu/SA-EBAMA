#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

metric_dir = Path("results/formal_main400_all_methods/metrics")

clip = pd.read_csv(metric_dir / "clip_metrics_with_storm.csv")
blip = pd.read_csv(metric_dir / "blip_tc_metrics_with_storm.csv")
runtime = pd.read_csv(metric_dir / "runtime_trace_metrics_with_storm.csv")

keys = ["method", "prompt_id", "seed"]

blip_keep = keys + ["tc_sim", "blip_caption"]
blip_keep = [c for c in blip_keep if c in blip.columns]

runtime_keep = keys + [
    "time_seconds",
    "peak_memory_mb",
    "num_attempted_steps",
    "num_updated_steps",
    "trace_total_loss_mean",
    "trace_sep_loss_mean",
    "oor",
]
runtime_keep = [c for c in runtime_keep if c in runtime.columns]

for c in runtime_keep:
    if c not in keys and c in clip.columns:
        clip = clip.drop(columns=[c])

for c in ["tc_sim", "blip_caption"]:
    if c in clip.columns:
        clip = clip.drop(columns=[c])

df = clip.merge(blip[blip_keep], on=keys, how="left")
df = df.merge(runtime[runtime_keep], on=keys, how="left")

if "weak_object_risk_proxy" not in df.columns:
    df["weak_object_risk_proxy"] = df["missing_proxy"]

out = metric_dir / "eval_all_metrics_with_storm_skip_t2i.csv"
df.to_csv(out, index=False)

print("saved:", out)
print(df.groupby("method")[[
    "clip_full",
    "clip_min",
    "clip_mean",
    "tc_sim",
    "weak_object_risk_proxy",
    "time_seconds",
]].mean())
