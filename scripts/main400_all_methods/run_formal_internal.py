#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SEEDS = [1, 42, 1234, 70690]

METHOD_CONFIGS: Dict[str, List[str]] = {
    "ebama": [
        "--no-use_separation",
        "--no-use_adaptive_guidance",
    ],
    "ebama_sep": [
        "--use_separation",
        "--gamma_sep", "0.05",
        "--sep_tau", "percentile",
        "--sep_temp", "0.15",
        "--sep_percentile", "0.70",
        "--no-use_adaptive_guidance",
    ],
    "ebama_adaptive": [
        "--no-use_separation",
        "--use_adaptive_guidance",
        "--adaptive_mode", "sparse10",
        "--max_guidance_step", "25",
        "--adaptive_overlap_threshold", "0.45",
        "--adaptive_dynamic_step",
    ],
    "sa_ebama": [
        "--use_separation",
        "--gamma_sep", "0.05",
        "--sep_tau", "percentile",
        "--sep_temp", "0.15",
        "--sep_percentile", "0.70",
        "--use_adaptive_guidance",
        "--adaptive_mode", "sparse10",
        "--max_guidance_step", "25",
        "--adaptive_overlap_threshold", "0.45",
        "--adaptive_dynamic_step",
    ],
}


def read_prompt_csv(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_prompt_dirs(method_out: Path, prompt_id: str):
    """Find output directories created by inference.py for a given prompt_id."""
    dirs = []
    dirs.extend(method_out.glob(f"{prompt_id}__*"))
    dirs.extend(method_out.glob(f"{prompt_id}_*"))
    dirs.extend(method_out.glob(f"{prompt_id}*"))

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for d in dirs:
        if d.is_dir() and d not in seen:
            unique.append(d)
            seen.add(d)
    return unique


def is_complete_output(method_out: Path, prompt_id: str, seed: int, min_image_bytes: int = 1024) -> bool:
    """
    A sample is considered complete only when both image and guidance trace exist.
    This avoids skipping failed or partially written runs.
    """
    prompt_dirs = find_prompt_dirs(method_out, prompt_id)
    if not prompt_dirs:
        return False

    for prompt_dir in prompt_dirs:
        image_candidates = list(prompt_dir.glob(f"seed{seed}_*.png"))
        trace_candidates = list(prompt_dir.glob(f"seed{seed}_*_guidance_trace.json"))

        valid_image = any(p.exists() and p.stat().st_size >= min_image_bytes for p in image_candidates)
        valid_trace = any(p.exists() and p.stat().st_size > 0 for p in trace_candidates)

        if valid_image and valid_trace:
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--csv", type=str, default="dataset/formal_prompts_main_400.csv")
    parser.add_argument("--out_root", type=str, default="outputs/formal_main400_internal_locked")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ebama", "ebama_sep", "ebama_adaptive", "sa_ebama"],
    )
    parser.add_argument("--step_size", type=float, default=20.0)
    parser.add_argument("--attn_res", type=int, default=256)
    parser.add_argument("--dist", type=str, default="cos")
    parser.add_argument("--lambda_ours", type=float, default=0.5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompt-seed pairs whose image and guidance trace already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force rerun even if outputs already exist. Overrides --resume.",
    )
    parser.add_argument(
        "--min_image_bytes",
        type=int,
        default=1024,
        help="Minimum PNG file size used when checking whether an output image is complete.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    csv_path = root / args.csv
    out_root = root / args.out_root
    inference_py = root / "inference.py"

    if not inference_py.exists():
        raise FileNotFoundError(f"inference.py not found at {inference_py}")

    if not csv_path.exists():
        raise FileNotFoundError(f"prompt csv not found at {csv_path}")

    records = read_prompt_csv(csv_path)
    records = records[args.start_idx:args.end_idx]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for method in args.methods:
        if method not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method: {method}. Choices: {list(METHOD_CONFIGS)}")

        method_out = out_root / method
        if not args.dry_run:
            method_out.mkdir(parents=True, exist_ok=True)

        for row in records:
            prompt_id = row["prompt_id"]
            prompt = row["prompt"]

            for seed in SEEDS:
                if args.resume and not args.overwrite:
                    if is_complete_output(method_out, prompt_id, seed, args.min_image_bytes):
                        print("=" * 100)
                        print(f"SKIP existing output: METHOD={method} PROMPT_ID={prompt_id} SEED={seed}")
                        continue

                cmd = [
                    sys.executable,
                    str(inference_py),
                    "--prompt", prompt,
                    "--prompt_id", prompt_id,
                    "--method_name", method,
                    "--seed", str(seed),
                    "--output_directory", str(method_out),
                    "--model_path", args.model_path,
                    "--gpu", "0",
                    "--step_size", str(args.step_size),
                    "--attn_res", str(args.attn_res),
                    "--dist", args.dist,
                    "--lambda_ours", str(args.lambda_ours),
                    "--force_ddim",
                    "--log_guidance",
                ]
                cmd.extend(METHOD_CONFIGS[method])

                print("=" * 100)
                print(f"METHOD={method} PROMPT_ID={prompt_id} SEED={seed}")
                print(" ".join(cmd))

                if not args.dry_run:
                    subprocess.run(cmd, cwd=str(root), env=env, check=True)

    print("All formal internal ablation jobs finished.")


if __name__ == "__main__":
    main()
