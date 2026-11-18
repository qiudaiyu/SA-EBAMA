#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#   sd
#   attend_excite
#   syngen
#   selfcross

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = [1, 42, 1234, 70690]


def parse_args():
    parser = argparse.ArgumentParser()

    # Basic paths
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--csv", type=str, default="dataset/formal_prompts_main_400.csv")
    parser.add_argument("--out_root", type=str, default="outputs/formal_main400_external_baselines")
    parser.add_argument("--model_path", type=str, required=True)

    # External repositories
    parser.add_argument("--repo_attend_excite", type=str, default="external/Attend-and-Excite")
    parser.add_argument("--repo_syngen", type=str, default="external/Linguistic-Binding-in-Diffusion-Models")
    parser.add_argument("--repo_conform", type=str, default="external/CONFORM")
    parser.add_argument("--repo_storm", type=str, default="external/STORM")

    # Methods
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sd", "attend_excite", "syngen"],
        help=(
            "External methods to run. Supported: "
            "sd, attend_excite, syngen, conform, storm"
        ),
    )

    # Shared generation config
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--scheduler", type=str, default="ddim")

    # Prompt slicing
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    # Resume / overwrite
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Optional command templates for older external wrappers
    parser.add_argument(
        "--attend_excite_cmd_template",
        type=str,
        default=None,
        help="Argument template passed to the Attend-and-Excite wrapper.",
    )
    parser.add_argument(
        "--syngen_cmd_template",
        type=str,
        default=None,
        help="Argument template passed to the SynGen wrapper.",
    )

    parser.add_argument("--conform_max_iter_to_alter", type=int, default=25)
    parser.add_argument("--conform_refinement_steps", type=int, default=20)
    parser.add_argument("--conform_scale_factor", type=float, default=20.0)
    parser.add_argument("--conform_temperature", type=float, default=0.5)
    parser.add_argument("--conform_loss_fn", type=str, default="ntxent")
    parser.add_argument("--conform_practical", action="store_true")
    parser.add_argument("--storm_max_iter_to_alter", type=int, default=25)
    parser.add_argument("--storm_attention_res", type=int, default=16)
    parser.add_argument("--storm_scale_factor", type=int, default=20)
    parser.add_argument("--storm_fp16", action="store_true")


    return parser.parse_args()


def add_common_args(cmd, args, *, storm=False):
    cmd.extend([
        "--csv", args.csv,
        "--model_path", args.model_path,
        "--seeds", *[str(s) for s in args.seeds],
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--gpu", str(args.gpu),
        "--start_idx", str(args.start_idx),
    ])

    if not storm:
        cmd.extend(["--height", str(args.height), "--width", str(args.width), "--scheduler", args.scheduler])
    if args.end_idx is not None:
        cmd.extend(["--end_idx", str(args.end_idx)])
    if args.resume:
        cmd.append("--resume")
    if args.overwrite:
        cmd.append("--overwrite")


def normalize_method_name(method: str) -> str:
    m = str(method).strip().lower().replace("-", "_")
    aliases = {
        "stable_diffusion": "sd",
        "attend_and_excite": "attend_excite",
        "ane": "attend_excite",
        "sg": "syngen",
        "conform_guidance": "conform",
        "storm_guidance": "storm",
    }
    return aliases.get(m, m)


def main():
    args = parse_args()

    root = Path(args.project_root).resolve()
    out_root = root / args.out_root
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    scripts_dir = root / "scripts" / "main400_all_methods"
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Baseline scripts not found: {scripts_dir}")

    methods = [normalize_method_name(m) for m in args.methods]

    for method in methods:
        if method == "sd":
            script = scripts_dir / "run_sd_baseline.py"
            cmd = [
                sys.executable,
                str(script),
                "--out_root", str(out_root / "sd"),
            ]
            add_common_args(cmd, args)

        elif method == "attend_excite":
            script = scripts_dir / "run_attend_excite_wrapper.py"
            cmd = [
                sys.executable,
                str(script),
                "--out_root", str(out_root / "attend_excite"),
                "--repo_root", args.repo_attend_excite,
            ]
            add_common_args(cmd, args)
            if args.attend_excite_cmd_template:
                cmd.extend(["--cmd_template", args.attend_excite_cmd_template])

        elif method == "syngen":
            script = scripts_dir / "run_syngen_wrapper.py"
            cmd = [
                sys.executable,
                str(script),
                "--out_root", str(out_root / "syngen"),
                "--repo_root", args.repo_syngen,
            ]
            add_common_args(cmd, args)
            if args.syngen_cmd_template:
                cmd.extend(["--cmd_template", args.syngen_cmd_template])

        elif method == "storm":
            script = scripts_dir / "run_storm_wrapper.py"
            cmd = [
                sys.executable,
                str(script),
                "--out_root", str(out_root / "storm"),
                "--repo_root", args.repo_storm,
                "--max_iter_to_alter", str(args.storm_max_iter_to_alter),
                "--attention_res", str(args.storm_attention_res),
                "--scale_factor", str(args.storm_scale_factor),
            ]
            add_common_args(cmd, args, storm=True)
            if args.storm_fp16:
                cmd.append("--fp16")

        elif method == "conform":
            script = scripts_dir / "run_conform_wrapper.py"
            cmd = [
                sys.executable,
                str(script),
                "--out_root", str(out_root / "conform"),
                "--repo_root", args.repo_conform,
                "--max_iter_to_alter", str(args.conform_max_iter_to_alter),
                "--refinement_steps", str(args.conform_refinement_steps),
                "--scale_factor", str(args.conform_scale_factor),
                "--temperature", str(args.conform_temperature),
                "--loss_fn", args.conform_loss_fn,
            ]
            add_common_args(cmd, args)
            if args.conform_practical:
                cmd.append("--practical")

        else:
            raise ValueError(
                f"Unknown external method: {method}. "
                "Supported methods: sd, attend_excite, syngen, conform, storm"
            )

        print("=" * 100)
        print(f"RUN EXTERNAL METHOD: {method}")
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, cwd=str(root), check=True)

    print("All selected external baselines finished.")


if __name__ == "__main__":
    main()
