#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

eval_csv = Path("results/formal_main400_all_methods/metrics/eval_all_metrics_with_storm_with_t2i.csv")
out_dir = Path("results/formal_main400_all_methods/tables/with_storm_with_t2i")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(eval_csv)

main_methods = ["sd", "attend_excite", "syngen", "conform", "storm", "ebama", "sa_ebama"]
method_name = {
    "sd": "SD",
    "attend_excite": "Attend-and-Excite",
    "syngen": "SynGen",
    "conform": "CONFORM",
    "storm": "STORM",
    "ebama": "EBAMA",
    "sa_ebama": "SA-EBAMA",
}

metrics = [
    "clip_full",
    "clip_min",
    "clip_mean",
    "tc_sim",
    "t2i_spatial",
    "weak_object_risk_proxy",
    "time_seconds",
]

rows = []
for m in main_methods:
    sub = df[df["method"] == m].copy()
    row = {
        "method": m,
        "Method": method_name[m],
        "N": int(len(sub)),
    }
    for col in metrics:
        if col in sub.columns:
            row[col] = float(sub[col].mean())
    rows.append(row)

table = pd.DataFrame(rows)

out = out_dir / "paper_table_main_comparison_with_storm_with_t2i.csv"
table.to_csv(out, index=False)

print("saved:", out)
print(table.to_string(index=False))
