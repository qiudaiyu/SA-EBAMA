#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np


def import_dvmp_module(dvmp_py: Path):
    spec = importlib.util.spec_from_file_location("syngen_dvmp", str(dvmp_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dvmp_py",
        type=str,
        default="external/Linguistic-Binding-in-Diffusion-Models/dvmp.py",
    )
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dest_path",
        type=str,
        default="data/raw/DVMP/dvmp_raw_1000_seed42.csv",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dvmp = import_dvmp_module(Path(args.dvmp_py))

    dest = Path(args.dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    dvmp.dvmp_dataset_creation(
        num_samples=args.num_samples,
        dest_path=str(dest),
    )

    print(f"Saved DVMP raw prompts to {dest}")


if __name__ == "__main__":
    main()
