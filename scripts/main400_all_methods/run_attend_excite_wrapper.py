#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from tqdm import tqdm

from formal_utils import (
    copy_or_move_image,
    ensure_dir,
    external_sample_complete,
    find_first_image,
    prompt_dir_name,
    read_prompt_csv,
    write_json,
)


DEFAULT_CMD_TEMPLATE = (
    "{python} {repo_root}/scripts/run.py "
    "--prompt \"{prompt}\" "
    "--seed {seed} "
    "--output_dir \"{tmp_dir}\" "
    "--pretrained_model_name_or_path \"{model_path}\" "
    "--num_inference_steps {num_inference_steps} "
    "--guidance_scale {guidance_scale} "
    "--height {height} "
    "--width {width}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--repo_root", type=str, required=True)
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
    parser.add_argument(
        "--cmd_template",
        type=str,
        default=DEFAULT_CMD_TEMPLATE,
        help=(
            "Attend-and-Excite argument template. Available fields: "
            "{python}, {repo_root}, {prompt}, {prompt_id}, {seed}, {tmp_dir}, "
            "{model_path}, {num_inference_steps}, {guidance_scale}, {height}, {width}, {scheduler}"
        ),
    )
    return parser.parse_args()


def guess_token_indices(prompt: str) -> str:
    """Estimate CLIP token positions for Attend-and-Excite."""


    stop = {
        "a", "an", "the", "and", "with", "of", "to", "on", "in",
        "left", "right", "above", "below", "under", "over",
        "front", "behind", "next", "beside", "near"
    }
    attrs = {
        "red", "blue", "green", "yellow", "black", "white", "purple",
        "orange", "pink", "brown", "gray", "grey", "striped", "spotted",
        "checkered", "wooden", "metal", "plastic", "glass", "small",
        "large", "shiny", "old", "modern"
    }

    words = prompt.replace(",", " ").replace(".", " ").split()
    indices = []
    for i, w in enumerate(words, start=1):
        clean = w.lower().strip(".,;:!?")
        if clean in stop or clean in attrs:
            continue
        indices.append(i)

    if len(indices) >= 2:
        indices = indices[:3]
    elif len(indices) == 1:
        indices = indices
    else:
        indices = [2]

    return "[" + ",".join(str(x) for x in indices) + "]"


def build_command(args, row, seed: int, tmp_dir: Path):
    prompt = row["prompt"]
    prompt_id = row["prompt_id"]

    local_model = Path(args.model_path)
    values = {
        "python": sys.executable,
        "repo_root": str(Path(args.repo_root).resolve()),
        "prompt": prompt,
        "prompt_id": prompt_id,
        "seed": int(seed),
        "tmp_dir": str(tmp_dir),
        "model_path": str(local_model.resolve()) if local_model.exists() else args.model_path,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "scheduler": args.scheduler,
        "token_indices": guess_token_indices(prompt),
    }
    return [part.format(**values) for part in shlex.split(args.cmd_template)]


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Attend-and-Excite repo not found: {repo_root}")

    csv_path = Path(args.csv)
    out_root = ensure_dir(Path(args.out_root))

    records = read_prompt_csv(csv_path)
    records = records[args.start_idx:args.end_idx]

    write_json(out_root / "run_metadata.json", {
        "method": "attend_excite",
        "csv": str(csv_path),
        "repo_root": str(repo_root),
        "model_path": args.model_path,
        "scheduler": args.scheduler,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seeds": args.seeds,
        "cmd_template": args.cmd_template,
    })

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for row in tqdm(records, desc="Attend-and-Excite prompts"):
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        dataset = row.get("dataset", "")
        prompt_dir = ensure_dir(out_root / prompt_dir_name(prompt_id, prompt))

        for seed in args.seeds:
            image_path = prompt_dir / f"seed{seed}.png"
            meta_path = prompt_dir / f"seed{seed}_metadata.json"
            log_path = prompt_dir / f"seed{seed}_external_stdout.log"

            if args.resume and not args.overwrite:
                if external_sample_complete(prompt_dir, seed, args.min_image_bytes):
                    continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with tempfile.TemporaryDirectory(prefix=f"ane_{prompt_id}_seed{seed}_") as td:
                tmp_dir = Path(td)
                cmd = build_command(args, row, seed, tmp_dir)

                start_time = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                elapsed = time.perf_counter() - start_time

                log_path.write_text(proc.stdout or "", encoding="utf-8")

                if proc.returncode != 0:
                    write_json(meta_path, {
                        "method": "attend_excite",
                        "external_baseline": True,
                        "status": "failed",
                        "returncode": proc.returncode,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "dataset": dataset,
                        "seed": int(seed),
                        "command": cmd,
                        "log_path": str(log_path),
                        "time_seconds": float(elapsed),
                    })
                    raise RuntimeError(
                        f"Attend-and-Excite failed: prompt_id={prompt_id}, seed={seed}. "
                        f"See log: {log_path}"
                    )

                generated = find_first_image(tmp_dir)
                if generated is None:
                    write_json(meta_path, {
                        "method": "attend_excite",
                        "external_baseline": True,
                        "status": "failed_no_image",
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "dataset": dataset,
                        "seed": int(seed),
                        "command": cmd,
                        "tmp_dir": str(tmp_dir),
                        "log_path": str(log_path),
                        "time_seconds": float(elapsed),
                    })
                    raise FileNotFoundError(
                        f"No generated image found for Attend-and-Excite: "
                        f"prompt_id={prompt_id}, seed={seed}"
                    )

                copy_or_move_image(generated, image_path, move=False)

            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else None
            )

            write_json(meta_path, {
                "method": "attend_excite",
                "external_baseline": True,
                "status": "ok",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "dataset": dataset,
                "seed": int(seed),
                "image_path": str(image_path),
                "time_seconds": float(elapsed),
                "peak_memory_mb": peak_memory_mb,
                "repo_root": str(repo_root),
                "model_path": args.model_path,
                "scheduler": args.scheduler,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "command": cmd,
                "log_path": str(log_path),
            })

    print("Attend-and-Excite baseline finished.")


if __name__ == "__main__":
    main()
