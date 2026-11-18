#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import BlipProcessor, BlipForConditionalGeneration
import clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blip_model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.index_csv)
    df = df[df["exists"] == 1].copy()

    print("Loading BLIP:", args.blip_model)
    processor = BlipProcessor.from_pretrained(args.blip_model)
    blip_model = BlipForConditionalGeneration.from_pretrained(args.blip_model).to(device)
    blip_model.eval()

    print("Loading CLIP:", args.clip_model)
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="BLIP + T-C Sim"):
        image_path = r["image_path"]
        prompt = str(r["prompt"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            row = r.to_dict()
            row.update({
                "blip_caption": "",
                "tc_sim": None,
                "error": str(e),
            })
            rows.append(row)
            continue

        with torch.no_grad():
            inputs = processor(image, return_tensors="pt").to(device)
            out = blip_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
            caption = processor.decode(out[0], skip_special_tokens=True)

            texts = clip.tokenize([prompt, caption], truncate=True).to(device)
            feats = clip_model.encode_text(texts)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            tc_sim = float((feats[0] @ feats[1]).item())

        row = r.to_dict()
        row.update({
            "blip_caption": caption,
            "tc_sim": tc_sim,
            "error": "",
        })
        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
