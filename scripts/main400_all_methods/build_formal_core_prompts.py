#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import importlib.util
import json
import random
import re
from pathlib import Path


COLOR_WORDS = [
    "red", "blue", "green", "yellow", "black", "white", "brown",
    "gray", "grey", "orange", "purple", "pink", "silver", "gold",
    "golden", "beige", "tan"
]

BAD_COLOR_WORDS = {"snow"}

SPATIAL_PATTERNS = [
    r"\s+to the left of\s+",
    r"\s+to the right of\s+",
    r"\s+on the left of\s+",
    r"\s+on the right of\s+",
    r"\s+on the top of\s+",
    r"\s+on the bottom of\s+",
    r"\s+next to\s+",
    r"\s+near\s+",
    r"\s+on side of\s+",
    r"\s+above\s+",
    r"\s+below\s+",
    r"\s+under\s+",
    r"\s+behind\s+",
    r"\s+in front of\s+",
    r"\s+beside\s+",
]


def load_py_var(path: Path, var_name: str):
    spec = importlib.util.spec_from_file_location("loaded_module", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, var_name):
        raise ValueError(f"{path} does not define variable `{var_name}`")
    return getattr(module, var_name)


def read_lines(path: Path):
    return [
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def add_article_if_needed(text: str) -> str:
    text = text.strip(" .,;:")
    if re.match(r"^(a|an|the|one|two|three|some)\s+", text, re.I):
        return text
    return "an " + text if text[:1].lower() in "aeiou" else "a " + text


def split_simple_subprompts(prompt: str):
    text = normalize_space(prompt)
    parts = re.split(
        r"\s+(?:and|with|beside|next to|near)\s+",
        text,
        flags=re.IGNORECASE,
    )
    parts = [add_article_if_needed(p) for p in parts if p.strip(" .,;:")]
    return parts if len(parts) >= 2 else [text.strip(" .")]


def split_spatial_subprompts(prompt: str):
    text = normalize_space(prompt)
    for pat in SPATIAL_PATTERNS:
        parts = re.split(pat, text, flags=re.IGNORECASE)
        if len(parts) == 2:
            return [add_article_if_needed(parts[0]), add_article_if_needed(parts[1])]
    return split_simple_subprompts(text)


def extract_color_phrases(prompt: str):
    lowered = prompt.lower()
    color_pattern = "|".join(re.escape(c) for c in COLOR_WORDS)

    raw_matches = re.finditer(
        rf"\b({color_pattern})\b(?:\s+[a-zA-Z][a-zA-Z\-]*){{1,3}}",
        lowered,
    )

    stop_words = {
        "and", "or", "with", "in", "on", "at", "near", "next",
        "beside", "behind", "front", "of", "to", "from", "while",
        "which", "that", "is", "are", "was", "were", "has", "have",
        "the", "a", "an"
    }

    phrases = []
    seen = set()

    for m in raw_matches:
        phrase = m.group(0).strip(" .,:;")
        words = phrase.split()

        if not words:
            continue

        color = words[0]
        if color in BAD_COLOR_WORDS:
            continue

        clean = []
        for i, w in enumerate(words):
            if i > 0 and w in stop_words:
                break
            clean.append(w)

        if len(clean) < 2:
            continue

        clean_phrase = " ".join(clean[:3])
        clean_phrase = add_article_if_needed(clean_phrase)

        key = re.sub(r"[^a-z0-9]+", "", clean_phrase.lower())
        if key not in seen:
            phrases.append(clean_phrase)
            seen.add(key)

    return phrases


def count_attributes(prompt: str) -> int:
    lowered = prompt.lower()
    return sum(
        1 for c in COLOR_WORDS
        if re.search(rf"\b{re.escape(c)}\b", lowered)
    )


def has_color(prompt: str) -> int:
    return int(count_attributes(prompt) > 0)


def make_record(dataset, prompt_id, prompt, subprompts, source, subtype, difficulty_tag, has_spatial):
    return {
        "dataset": dataset,
        "split": "formal",
        "prompt_id": prompt_id,
        "prompt": normalize_space(prompt),
        "subprompts_json": json.dumps(subprompts, ensure_ascii=False),
        "object_count": len(subprompts),
        "attribute_count": count_attributes(prompt),
        "source": source,
        "subtype": subtype,
        "has_color": has_color(prompt),
        "has_spatial": int(has_spatial),
        "difficulty_tag": difficulty_tag,
    }


def allocate_counts(total, keys):
    base = total // len(keys)
    rem = total % len(keys)
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def build_ane(ane_py: Path, n: int, rng):
    data = load_py_var(ane_py, "data")
    subtype_map = {
        "animals": "animal_animal",
        "animals_objects": "animal_object",
        "objects": "object_object",
    }

    available_keys = [k for k in subtype_map if k in data and len(data[k]) > 0]
    quotas = allocate_counts(n, available_keys)

    rows = []
    for key in available_keys:
        subtype = subtype_map[key]
        prompts = list(data.get(key, []))
        rng.shuffle(prompts)

        added = 0
        for prompt in prompts:
            if added >= quotas[key]:
                break

            subprompts = split_simple_subprompts(prompt)
            if len(subprompts) < 2:
                continue

            rows.append(
                make_record(
                    "AnE",
                    f"ane_{subtype}_{len(rows):04d}",
                    prompt,
                    subprompts,
                    "ane_data.py",
                    subtype,
                    "two_object",
                    0,
                )
            )
            added += 1

    return rows[:n]


def build_abc(abc_subset_py: Path, abc_txt: Path, n: int, rng):
    if abc_subset_py.exists():
        prompts = list(load_py_var(abc_subset_py, "abc"))
        source = "data_abc.py"
    elif abc_txt.exists():
        prompts = read_lines(abc_txt)
        source = "ABC-6K.txt"
    else:
        raise FileNotFoundError("Neither data_abc.py nor ABC-6K.txt was found.")

    rng.shuffle(prompts)
    rows = []

    for prompt in prompts:
        subprompts = extract_color_phrases(prompt)

        if len(subprompts) < 2:
            subprompts = split_simple_subprompts(prompt)

        if len(subprompts) < 2:
            continue

        rows.append(
            make_record(
                "ABC-6K-subset",
                f"abc_{len(rows):04d}",
                prompt,
                subprompts,
                source,
                "color_caption",
                "natural_caption",
                0,
            )
        )

        if len(rows) >= n:
            break

    return rows


def build_t2i_spatial(spatial_txt: Path, n: int, rng):
    prompts = read_lines(spatial_txt)
    rng.shuffle(prompts)

    rows = []
    for prompt in prompts:
        subprompts = split_spatial_subprompts(prompt)
        if len(subprompts) < 2:
            continue

        rows.append(
            make_record(
                "T2I-Spatial",
                f"t2i_spatial_{len(rows):04d}",
                prompt,
                subprompts,
                "T2I-CompBench spatial_val.txt",
                "spatial",
                "spatial",
                1,
            )
        )

        if len(rows) >= n:
            break

    return rows


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset", "split", "prompt_id", "prompt", "subprompts_json",
        "object_count", "attribute_count", "source", "subtype",
        "has_color", "has_spatial", "difficulty_tag",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ane_py", type=str, required=True)
    parser.add_argument("--abc_subset_py", type=str, required=True)
    parser.add_argument("--abc_txt", type=str, required=True)
    parser.add_argument("--t2i_spatial_txt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="formal_prompts")
    parser.add_argument("--n_ane", type=int, default=100)
    parser.add_argument("--n_abc", type=int, default=100)
    parser.add_argument("--n_t2i", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ane_rows = build_ane(Path(args.ane_py), args.n_ane, rng)
    abc_rows = build_abc(Path(args.abc_subset_py), Path(args.abc_txt), args.n_abc, rng)
    t2i_rows = build_t2i_spatial(Path(args.t2i_spatial_txt), args.n_t2i, rng)

    save_csv(ane_rows, out_dir / "ane_formal_100.csv")
    save_csv(abc_rows, out_dir / "abc_formal_100.csv")
    save_csv(t2i_rows, out_dir / "t2i_spatial_formal_100.csv")

    summary = {
        "AnE": len(ane_rows),
        "ABC-6K-subset": len(abc_rows),
        "T2I-Spatial": len(t2i_rows),
    }
    (out_dir / "core_formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
