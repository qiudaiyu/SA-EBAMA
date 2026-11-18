#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "gray", "black", "white", "beige", "teal",
}

MATERIAL_TEXTURE_WORDS = {
    "furry", "baby", "spotted", "sleepy", "modern", "wooden",
    "metal", "curved", "spiky", "checkered", "sliced", "skewered",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def parse_subjects(x):
    if isinstance(x, list):
        return x
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return []


def build_subprompts(prompt: str, subjects):
    """Split a DVMP prompt into phrases, falling back to the supplied subjects."""


    prompt = normalize_space(prompt)
    parts = re.split(r"\s+and\s+", prompt, flags=re.IGNORECASE)
    parts = [p.strip(" .,:;") for p in parts if p.strip(" .,:;")]

    if len(parts) >= 1:
        return parts

    return [f"a {s}" for s in subjects]


def count_attributes(prompt: str) -> int:
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    return sum(1 for w in words if w in COLOR_WORDS or w in MATERIAL_TEXTURE_WORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/formal_prompts/dvmp_formal.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.raw_csv)
    if "prompt" not in df.columns:
        raise ValueError("DVMP raw csv must contain a 'prompt' column.")

    df = df.drop_duplicates(subset=["prompt"]).sample(
        n=min(args.n, len(df)),
        random_state=args.seed,
    ).reset_index(drop=True)

    rows = []
    for i, row in df.iterrows():
        prompt = normalize_space(row["prompt"])
        subjects = parse_subjects(row.get("subjects", "[]"))
        subprompts = build_subprompts(prompt, subjects)

        rows.append({
            "dataset": "DVMP",
            "split": "formal",
            "prompt_id": f"dvmp_{i:04d}",
            "prompt": prompt,
            "subprompts_json": json.dumps(subprompts, ensure_ascii=False),
            "object_count": len(subprompts),
            "attribute_count": int(row.get("num_modifiers", count_attributes(prompt))),
            "source": "SynGen official dvmp.py",
            "subtype": f"num_phrases_{int(row.get('num_phrases', len(subprompts)))}",
            "has_color": int(any(c in prompt.lower().split() for c in COLOR_WORDS)),
            "has_spatial": 0,
            "difficulty_tag": "multi_attribute",
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")

    print(f"Saved {len(rows)} DVMP formal prompts to {out}")


if __name__ == "__main__":
    main()
