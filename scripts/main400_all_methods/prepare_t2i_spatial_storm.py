#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import pandas as pd

index_csv = "results/formal_main400_all_methods/all_images_index_with_storm.csv"
out_root = Path("results/formal_main400_all_methods/t2i_spatial_prepare/storm")
out_root.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(index_csv)
sub = df[
    (df["method"] == "storm") &
    (df["dataset"] == "T2I-Spatial") &
    (df["exists"] == 1)
].copy()

for _, r in sub.iterrows():
    d = out_root / f'{r["prompt_id"]}__seed{r["seed"]}'
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "image.png"
    if not dst.exists():
        shutil.copy2(r["image_path"], dst)

sub.to_csv("results/formal_main400_all_methods/t2i_spatial_prepare/storm_index.csv", index=False)

print("prepared images:", len(sub))
print("saved index: results/formal_main400_all_methods/t2i_spatial_prepare/storm_index.csv")
