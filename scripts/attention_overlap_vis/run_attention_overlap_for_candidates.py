

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd):
    print("\n" + " ".join([str(x) for x in cmd]) + "\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_root", default="results/attention_overlap_vis/rerun_screened")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vis_steps", default="0,1,2,4,6,8,10,12,16,20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.candidate_csv)
    if args.limit is not None:
        df = df.head(args.limit)

    out_root = Path(args.out_root)

    for _, row in df.iterrows():
        prompt_id = str(row["prompt_id"])
        prompt = str(row["prompt"])
        seed = int(row["seed"])

        jobs = [
            {
                "method_name": "ebama",
                "out_dir": out_root / "ebama",
                "args": [
                    "--no-use_separation",
                    "--gamma_sep", "0.0",
                    "--sep_tau", "percentile",
                    "--sep_percentile", "0.70",
                    "--sep_temp", "0.15",
                    "--no-use_adaptive_guidance",
                    "--adaptive_mode", "fixed25",
                    "--max_guidance_step", "25",
                    "--no-adaptive_dynamic_step",
                ],
            },
            {
                "method_name": "sa_ebama",
                "out_dir": out_root / "sa_ebama",
                "args": [
                    "--use_separation",
                    "--gamma_sep", "0.05",
                    "--sep_tau", "percentile",
                    "--sep_percentile", "0.70",
                    "--sep_temp", "0.15",
                    "--use_adaptive_guidance",
                    "--adaptive_mode", "sparse10",
                    "--adaptive_overlap_threshold", "0.45",
                    "--adaptive_dynamic_step",
                    "--max_guidance_step", "25",
                ],
            },
        ]

        for job in jobs:
            method = job["method_name"]
            out_dir = job["out_dir"]

            if args.skip_existing:
                found = list(out_dir.glob(f"*{prompt_id}*/seed{seed}_alpha20_overlap_vis/avg_overlap_heatmap.npy"))
                if found:
                    print(f"[SKIP] {method} {prompt_id} seed{seed}: {found[0]}")
                    continue

            cmd = [
                sys.executable, "-u", "inference.py",
                "--prompt", prompt,
                "--prompt_id", prompt_id,
                "--method_name", method,
                "--seed", str(seed),
                "--output_directory", str(out_dir),
                "--model_path", args.model_path,
                "--step_size", "20",
                "--attn_res", "256",
                "--gpu", str(args.gpu),
                "--dist", "cos",
                "--lambda_ours", "0.5",
                "--save_overlap_vis",
                "--overlap_vis_steps", args.vis_steps,
            ] + job["args"]

            run_cmd(cmd)


if __name__ == "__main__":
    main()
