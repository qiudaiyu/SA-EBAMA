#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path

import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def parse_subprompts(x, prompt):
    try:
        v = json.loads(x)
        if isinstance(v, list) and len(v) > 0:
            return [str(a) for a in v]
    except Exception:
        pass
    return [prompt]


def encode_texts(model, texts, device):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    df = pd.read_csv(args.index_csv)
    df = df[df["exists"] == 1].copy()

    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="CLIP eval"):
        image_path = r["image_path"]
        prompt = str(r["prompt"])
        subprompts = parse_subprompts(r.get("subprompts_json", ""), prompt)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            rows.append({
                **r.to_dict(),
                "clip_full": None,
                "clip_min": None,
                "clip_mean": None,
                "missing_proxy": None,
                "error": str(e),
            })
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(image_input)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        full_text_feat = encode_texts(model, [prompt], device)
        full_sim = float((image_feat @ full_text_feat.T).item())

        sub_text_feat = encode_texts(model, subprompts, device)
        sub_sims = (image_feat @ sub_text_feat.T).squeeze(0).detach().cpu().tolist()

        clip_min = float(min(sub_sims))
        clip_mean = float(sum(sub_sims) / len(sub_sims))


        # This score gap is a proxy for uneven phrase alignment.
        missing_proxy = float(clip_mean - clip_min)

        rows.append({
            **r.to_dict(),
            "clip_full": full_sim,
            "clip_min": clip_min,
            "clip_mean": clip_mean,
            "missing_proxy": missing_proxy,
            "subprompt_sims_json": json.dumps(sub_sims),
            "error": "",
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved CLIP metrics to {out}")


if __name__ == "__main__":
    main()
