#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


INTERNAL_METHODS = {"ebama", "ebama_sep", "ebama_adaptive", "sa_ebama"}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_float(x):
    try:
        if x == "" or x is None:
            return None
        return float(x)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def extract_trace_metrics(trace_path):
    if not isinstance(trace_path, str) or not trace_path or not Path(trace_path).exists():
        return {
            "num_attempted_steps": None,
            "num_updated_steps": None,
            "trace_total_loss_mean": None,
            "trace_sep_loss_mean": None,
            "oor": None,
        }

    payload = read_json(trace_path)

    trace = payload.get("trace", None)
    if trace is None:
        trace = payload.get("guidance_trace", None)
    if trace is None:
        trace = payload.get("steps", [])

    if not isinstance(trace, list):
        trace = []

    attempted = payload.get("num_attempted_steps", None)
    updated = payload.get("num_updated_steps", None)

    if attempted is None:
        attempted = sum(
            1 for t in trace
            if isinstance(t, dict) and (
                t.get("attempted", False) or
                t.get("do_guidance", False)
            )
        )

    if updated is None:
        updated = sum(
            1 for t in trace
            if isinstance(t, dict) and (
                t.get("updated", False) or
                t.get("latent_updated", False)
            )
        )

    total_losses = []
    sep_losses = []
    overlaps = []

    for t in trace:
        if not isinstance(t, dict):
            continue

        candidates = [t]
        for key in ["loss", "loss_dict", "metrics"]:
            if isinstance(t.get(key), dict):
                candidates.append(t[key])

        for d in candidates:
            for k in ["total", "total_loss", "loss_total"]:
                if k in d:
                    total_losses.append(safe_float(d.get(k)))

            for k in ["sep", "loss_sep", "sep_loss", "separation_loss"]:
                if k in d:
                    sep_losses.append(safe_float(d.get(k)))

            for k in ["overlap", "oor", "object_overlap", "mean_overlap"]:
                if k in d:
                    overlaps.append(safe_float(d.get(k)))

    return {
        "num_attempted_steps": attempted,
        "num_updated_steps": updated,
        "trace_total_loss_mean": mean(total_losses),
        "trace_sep_loss_mean": mean(sep_losses),
        "oor": mean(overlaps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Collect runtime/trace"):
        row = r.to_dict()

        meta_path = row.get("metadata_path", "")
        trace_path = row.get("trace_path", "")

        meta = read_json(meta_path) if isinstance(meta_path, str) and meta_path else {}

        row["time_seconds"] = safe_float(row.get("time_seconds")) or safe_float(meta.get("time_seconds"))
        row["peak_memory_mb"] = safe_float(row.get("peak_memory_mb")) or safe_float(meta.get("peak_memory_mb"))

        if row.get("method") in INTERNAL_METHODS:
            row.update(extract_trace_metrics(trace_path))
        else:
            row["num_attempted_steps"] = 0
            row["num_updated_steps"] = 0
            row["trace_total_loss_mean"] = None
            row["trace_sep_loss_mean"] = None
            row["oor"] = None

        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
