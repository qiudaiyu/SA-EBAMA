#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from formal_utils import (
    ensure_dir,
    external_sample_complete,
    prompt_dir_name,
    read_prompt_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 1234, 70690])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--scheduler", type=str, default="ddim")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_image_bytes", type=int, default=1024)
    parser.add_argument("--disable_safety_checker", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_root = ensure_dir(Path(args.out_root))

    records = read_prompt_csv(csv_path)
    records = records[args.start_idx:args.end_idx]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe_kwargs = {"torch_dtype": dtype}
    if args.disable_safety_checker:
        pipe_kwargs.update({
            "safety_checker": None,
            "requires_safety_checker": False,
        })

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        **pipe_kwargs,
    ).to(device)

    if args.scheduler.lower() == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    write_json(out_root / "run_metadata.json", {
        "method": "sd",
        "csv": str(csv_path),
        "model_path": args.model_path,
        "scheduler": pipe.scheduler.__class__.__name__,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seeds": args.seeds,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
    })

    for row in tqdm(records, desc="SD prompts"):
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        dataset = row.get("dataset", "")
        prompt_dir = ensure_dir(out_root / prompt_dir_name(prompt_id, prompt))

        for seed in args.seeds:
            image_path = prompt_dir / f"seed{seed}.png"
            meta_path = prompt_dir / f"seed{seed}_metadata.json"

            if args.resume and not args.overwrite:
                if external_sample_complete(prompt_dir, seed, args.min_image_bytes):
                    continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            generator = torch.Generator(device=device).manual_seed(int(seed))
            start_time = time.perf_counter()

            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                ).images[0]

            elapsed = time.perf_counter() - start_time
            peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if torch.cuda.is_available()
                else None
            )

            image.save(image_path)

            write_json(meta_path, {
                "method": "sd",
                "external_baseline": True,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "dataset": dataset,
                "seed": int(seed),
                "image_path": str(image_path),
                "time_seconds": float(elapsed),
                "peak_memory_mb": peak_memory_mb,
                "model_path": args.model_path,
                "scheduler": pipe.scheduler.__class__.__name__,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
            })

    print("SD baseline finished.")


if __name__ == "__main__":
    main()
