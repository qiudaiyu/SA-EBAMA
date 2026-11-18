

import json
from pathlib import Path
import pandas as pd
import numpy as np

INDEX_CSV = "results/formal_main400_all_methods/all_images_index.csv"
OUT_CSV = "results/attention_overlap_vis/candidates/formal_trace_overlap_screen.csv"

VIS_STEPS = {0, 1, 2, 4, 6, 8, 10, 12, 16, 20}

df = pd.read_csv(INDEX_CSV)
df["method_norm"] = df["method"].astype(str).str.strip().str.lower()


df = df[df["method_norm"].isin(["ebama", "sa_ebama"])].copy()

if "exists" in df.columns:
    df = df[df["exists"].astype(int) == 1].copy()

if "object_count" in df.columns:
    df = df[df["object_count"].fillna(0).astype(int) >= 2].copy()

project_root = Path.cwd()

def resolve_path(p):
    p = Path(str(p).strip())
    if not p.is_absolute():
        p = project_root / p
    return p

def read_mean_overlap(trace_path):
    p = resolve_path(trace_path)
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    trace = data.get("trace", [])
    vals = []
    for item in trace:
        if "overlap" not in item:
            continue
        step = item.get("step", None)
        if step is not None and int(step) not in VIS_STEPS:
            continue
        try:
            vals.append(float(item["overlap"]))
        except Exception:
            pass

    if not vals:
        return None
    return float(np.mean(vals))

rows = []
for _, r in df.iterrows():
    trace_path = r.get("trace_path", "")
    mean_overlap = read_mean_overlap(trace_path)
    if mean_overlap is None:
        continue

    rows.append({
        "dataset": r["dataset"],
        "prompt_id": r["prompt_id"],
        "prompt": r["prompt"],
        "seed": int(r["seed"]),
        "method": r["method_norm"],
        "image_path": r["image_path"],
        "trace_path": trace_path,
        "trace_mean_overlap": mean_overlap,
        "object_count": r.get("object_count", ""),
        "attribute_count": r.get("attribute_count", ""),
        "has_spatial": r.get("has_spatial", ""),
        "has_color": r.get("has_color", ""),
    })

t = pd.DataFrame(rows)

pairs = []
for keys, sub in t.groupby(["dataset", "prompt_id", "prompt", "seed"]):
    eb = sub[sub["method"] == "ebama"]
    sa = sub[sub["method"] == "sa_ebama"]
    if eb.empty or sa.empty:
        continue

    eb_o = float(eb.iloc[0]["trace_mean_overlap"])
    sa_o = float(sa.iloc[0]["trace_mean_overlap"])
    delta = eb_o - sa_o

    pairs.append({
        "dataset": keys[0],
        "prompt_id": keys[1],
        "prompt": keys[2],
        "seed": int(keys[3]),
        "trace_ebama_overlap": eb_o,
        "trace_sa_overlap": sa_o,
        "trace_delta_overlap": delta,
        "trace_relative_drop": delta / eb_o if eb_o > 0 else np.nan,
        "ebama_image": eb.iloc[0]["image_path"],
        "sa_image": sa.iloc[0]["image_path"],
        "object_count": eb.iloc[0].get("object_count", ""),
        "attribute_count": eb.iloc[0].get("attribute_count", ""),
        "has_spatial": eb.iloc[0].get("has_spatial", ""),
        "has_color": eb.iloc[0].get("has_color", ""),
    })

out = pd.DataFrame(pairs)
out = out.sort_values(["trace_delta_overlap", "trace_relative_drop"], ascending=False)

Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(out.head(30)[[
    "dataset", "prompt_id", "prompt", "seed",
    "trace_ebama_overlap", "trace_sa_overlap",
    "trace_delta_overlap", "trace_relative_drop"
]].to_string(index=False))
