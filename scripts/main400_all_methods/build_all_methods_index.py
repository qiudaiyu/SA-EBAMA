#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from formal_utils import read_prompt_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--internal_root", type=str, default="outputs/formal_main400_internal_locked")
    parser.add_argument("--external_root", type=str, default="outputs/formal_main400_external_baselines")
    parser.add_argument("--out_csv", type=str, default="results/formal_main400_all_methods/all_images_index_with_storm.csv")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sd", "attend_excite", "syngen", "conform", "storm", "ebama", "ebama_sep", "ebama_adaptive", "sa_ebama"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 1234, 70690])
    return parser.parse_args()


def safe_glob_prompt_dir(method_root: Path, prompt_id: str) -> List[Path]:
    candidates = []
    candidates.extend(method_root.glob(f"{prompt_id}__*"))
    candidates.extend(method_root.glob(f"{prompt_id}_*"))
    candidates.extend(method_root.glob(f"{prompt_id}*"))
    seen = set()
    out = []
    for p in candidates:
        if p.is_dir() and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_external_sample(method_root: Path, prompt_id: str, seed: int):
    for d in safe_glob_prompt_dir(method_root, prompt_id):
        image = d / f"seed{seed}.png"
        meta = d / f"seed{seed}_metadata.json"
        if image.exists():
            return image, meta if meta.exists() else None
    return None, None


def find_internal_sample(method_root: Path, prompt_id: str, seed: int):
    for d in safe_glob_prompt_dir(method_root, prompt_id):
        image_candidates = sorted(d.glob(f"seed{seed}_*.png"))
        trace_candidates = sorted(d.glob(f"seed{seed}_*_guidance_trace.json"))

        image_candidates = [
            p for p in image_candidates
            if not p.name.endswith("_attention.png")
        ]

        if image_candidates:
            image = image_candidates[0]
            trace = trace_candidates[0] if trace_candidates else None
            return image, trace
    return None, None


def main():
    args = parse_args()

    records = read_prompt_csv(Path(args.csv))
    prompt_by_id = {r["prompt_id"]: r for r in records}

    external_methods = {"sd", "attend_excite", "syngen", "conform", "storm"}
    internal_methods = {"ebama", "ebama_sep", "ebama_adaptive", "sa_ebama"}

    rows = []

    for method in args.methods:
        if method in external_methods:
            method_root = Path(args.external_root) / method
        elif method in internal_methods:
            method_root = Path(args.internal_root) / method
        else:
            raise ValueError(f"Unknown method: {method}")

        for prompt_id, row in prompt_by_id.items():
            for seed in args.seeds:
                if method in external_methods:
                    image_path, meta_or_trace = find_external_sample(method_root, prompt_id, seed)
                    meta = load_json(meta_or_trace) if meta_or_trace else {}
                    trace_path = ""
                    metadata_path = str(meta_or_trace) if meta_or_trace else ""
                else:
                    image_path, meta_or_trace = find_internal_sample(method_root, prompt_id, seed)
                    meta = load_json(meta_or_trace) if meta_or_trace else {}
                    trace_path = str(meta_or_trace) if meta_or_trace else ""
                    metadata_path = ""

                rows.append({
                    "method": method,
                    "dataset": row.get("dataset", ""),
                    "split": row.get("split", ""),
                    "prompt_id": prompt_id,
                    "prompt": row.get("prompt", ""),
                    "subprompts_json": row.get("subprompts_json", ""),
                    "object_count": row.get("object_count", ""),
                    "attribute_count": row.get("attribute_count", ""),
                    "source": row.get("source", ""),
                    "subtype": row.get("subtype", ""),
                    "has_color": row.get("has_color", ""),
                    "has_spatial": row.get("has_spatial", ""),
                    "difficulty_tag": row.get("difficulty_tag", ""),
                    "seed": seed,
                    "image_path": str(image_path) if image_path else "",
                    "metadata_path": metadata_path,
                    "trace_path": trace_path,
                    "exists": int(image_path is not None and image_path.exists()),
                    "time_seconds": meta.get("time_seconds", ""),
                    "peak_memory_mb": meta.get("peak_memory_mb", ""),
                })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    missing = sum(1 for r in rows if not r["exists"])
    print(f"Saved index to {out_csv}")
    print(f"Total rows: {total}")
    print(f"Missing rows: {missing}")


if __name__ == "__main__":
    main()
