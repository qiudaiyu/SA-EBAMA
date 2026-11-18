#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from pathlib import Path


ANIMALS = [
    "dog", "cat", "bear", "rabbit", "horse", "elephant", "monkey",
    "frog", "turtle", "bird", "mouse", "lion",
]

OBJECTS = [
    "backpack", "car", "chair", "suitcase", "crown", "bench",
    "clock", "camera", "umbrella", "guitar", "bicycle", "bowl",
    "bottle", "cup", "table", "lamp", "helmet", "shoe",
]

FRUITS = [
    "apple", "banana", "strawberry", "tomato",
]

COLORS = [
    "red", "blue", "green", "yellow", "black", "white",
    "purple", "orange", "pink", "brown", "gray",
]

TEXTURES = [
    "striped", "spotted", "checkered",
]

MATERIALS = [
    "wooden", "metal", "plastic", "glass",
]

SIZES = [
    "small", "large", "shiny", "old", "modern",
]

RELATIONS = [
    ("to the left of", "left_right"),
    ("to the right of", "left_right"),
    ("above", "vertical"),
    ("below", "vertical"),
    ("under", "vertical"),
    ("over", "vertical"),
    ("in front of", "front_behind"),
    ("behind", "front_behind"),
    ("next to", "near"),
    ("beside", "near"),
]


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def sample_object(rng):
    category = rng.choice(["animal", "object", "fruit"])

    if category == "animal":
        return rng.choice(ANIMALS), category
    if category == "fruit":
        return rng.choice(FRUITS), category
    return rng.choice(OBJECTS), category


def sample_attribute(category, rng):
    if category == "animal":
        candidates = COLORS + TEXTURES + ["small", "large"]
    elif category == "fruit":
        candidates = COLORS + ["small", "large", "shiny"]
    else:
        candidates = COLORS + TEXTURES + MATERIALS + SIZES

    return rng.choice(candidates)


def build_prompt(obj1, cat1, obj2, cat2, relation, rng):
    attr1 = sample_attribute(cat1, rng)
    attr2 = sample_attribute(cat2, rng)


    tries = 0
    while attr2 == attr1 and tries < 10:
        attr2 = sample_attribute(cat2, rng)
        tries += 1

    phrase1 = f"{article(attr1)} {attr1} {obj1}"
    phrase2 = f"{article(attr2)} {attr2} {obj2}"
    prompt = f"{phrase1} {relation} {phrase2}"

    return prompt, phrase1, phrase2, attr1, attr2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/formal_prompts/attribute_spatial_1k.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    seen = set()

    while len(rows) < args.n:
        obj1, cat1 = sample_object(rng)
        obj2, cat2 = sample_object(rng)

        if obj1 == obj2:
            continue

        relation, relation_type = rng.choice(RELATIONS)
        prompt, phrase1, phrase2, attr1, attr2 = build_prompt(
            obj1, cat1, obj2, cat2, relation, rng
        )

        if prompt in seen:
            continue
        seen.add(prompt)

        idx = len(rows)
        rows.append({
            "dataset": "Attribute-Spatial-1K" if args.n >= 1000 else "Attribute-Spatial",
            "split": "formal",
            "prompt_id": f"attr_spatial_{idx:04d}",
            "prompt": prompt,
            "subprompts_json": json.dumps([phrase1, phrase2], ensure_ascii=False),
            "object_count": 2,
            "attribute_count": 2,
            "source": f"synthetic_attribute_spatial_seed{args.seed}",
            "subtype": relation_type,
            "has_color": int(attr1 in COLORS or attr2 in COLORS),
            "has_spatial": 1,
            "difficulty_tag": "attribute_spatial",
            "relation": relation,
            "object1": obj1,
            "object2": obj2,
            "attribute1": attr1,
            "attribute2": attr2,
            "object1_category": cat1,
            "object2_category": cat2,
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
