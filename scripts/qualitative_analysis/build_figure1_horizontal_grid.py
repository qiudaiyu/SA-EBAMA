#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 horizontal qualitative grid builder.

Purpose:
- Only for Figure 1 overall qualitative comparison.
- Layout: prompts as rows, methods as columns.
- Left side: prompt labels.
- Top side: method labels.
- Ours / SA-EBAMA is rendered in bold by default.

This script reuses robust image lookup, font loading, image loading,
and text fitting utilities from build_compact_qualitative_grid_v6.py.
"""

import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


METHOD_ALIASES = {
    "stable_diffusion": "sd",
    "stable-diffusion": "sd",
    "stable diffusion": "sd",
    "sd": "sd",

    "attend-and-excite": "attend_excite",
    "attend_and_excite": "attend_excite",
    "attend excite": "attend_excite",
    "attend_excite": "attend_excite",
    "ane": "attend_excite",

    "sg": "syngen",
    "syn_gen": "syngen",
    "syngen": "syngen",

    "conform": "conform",
    "con_form": "conform",
    "confrom": "conform",

    "storm": "storm",
    "spatial_transport": "storm",
    "spatial-transport": "storm",

    "ebama": "ebama",

    "sa-ebama": "sa_ebama",
    "sa_ebama": "sa_ebama",
    "saebama": "sa_ebama",
    "ours": "sa_ebama",
}

METHOD_LABELS = {
    "sd": "SD",
    "attend_excite": "AnE",
    "syngen": "SynGen",
    "conform": "CONFORM",
    "storm": "STORM",
    "ebama": "EBAMA",
    "sa_ebama": "Ours",
}


def norm_method(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return METHOD_ALIASES.get(s, s)


def import_v6(base_grid_py):
    p = Path(base_grid_py)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find base grid script: {p}")

    spec = importlib.util.spec_from_file_location("grid_v6", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_display_prompt(row, prompt_text_col=None):
    if prompt_text_col and prompt_text_col in row.index:
        val = row[prompt_text_col]
        if not pd.isna(val) and str(val).strip():
            return str(val)

    for c in ["display_prompt", "short_prompt", "prompt_label", "paper_prompt"]:
        if c in row.index:
            val = row[c]
            if not pd.isna(val) and str(val).strip():
                return str(val)

    return str(row["prompt"])


def get_lookup_prompt(row):
    if "full_prompt" in row.index:
        val = row["full_prompt"]
        if not pd.isna(val) and str(val).strip():
            return str(val)
    return str(row["prompt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_grid_py", default=str(Path(__file__).with_name("build_compact_qualitative_grid.py")))
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", default=None)
    ap.add_argument("--report_csv", required=True)

    # Figure 1 horizontal layout controls.
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--col_gap", type=int, default=12)
    ap.add_argument("--row_gap", type=int, default=10)
    ap.add_argument("--prompt_label_w", type=int, default=220)
    ap.add_argument("--method_header_h", type=int, default=70)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--header_gap", type=int, default=12)

    # Text controls.
    ap.add_argument("--prompt_text_col", default=None)
    ap.add_argument("--prompt_font_size", type=int, default=22)
    ap.add_argument("--prompt_min_font_size", type=int, default=14)
    ap.add_argument("--method_font_size", type=int, default=24)
    ap.add_argument("--method_min_font_size", type=int, default=16)
    ap.add_argument("--caption_spacing", type=int, default=4)
    ap.add_argument("--bold_methods", nargs="*", default=["sa_ebama", "ours"])
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--reject_near_black", action="store_true")
    args = ap.parse_args()

    v6 = import_v6(args.base_grid_py)

    index_df = v6.load_index(args.index_csv, args.project_root)
    manifest = pd.read_csv(args.manifest_csv)

    if "figure" in manifest.columns:
        cases = manifest[manifest["figure"].eq(args.figure)].copy()
        if cases.empty:
            raise ValueError(f"No rows for figure={args.figure} in {args.manifest_csv}")
    else:
        cases = manifest.copy()

    if "row_order" in cases.columns:
        cases = cases.sort_values("row_order")
    cases = cases.reset_index(drop=True)

    methods = [norm_method(m) for m in args.methods]
    bold_methods = {norm_method(m) for m in args.bold_methods}

    n_rows = len(cases)
    n_cols = len(methods)

    W = (
        args.outer
        + args.prompt_label_w
        + args.header_gap
        + n_cols * args.thumb
        + (n_cols - 1) * args.col_gap
        + args.outer
    )
    H = (
        args.outer
        + args.method_header_h
        + args.header_gap
        + n_rows * args.thumb
        + (n_rows - 1) * args.row_gap
        + args.outer
    )

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x_grid0 = args.outer + args.prompt_label_w + args.header_gap
    y_grid0 = args.outer + args.method_header_h + args.header_gap

    method_font = v6.load_font(args.method_font_size, serif=True, bold=False)
    method_font_bold = v6.load_font(args.method_font_size, serif=True, bold=True)

    # Draw method names on the top.
    for ci, method in enumerate(methods):
        x0 = x_grid0 + ci * (args.thumb + args.col_gap)
        y0 = args.outer
        x1 = x0 + args.thumb
        y1 = args.outer + args.method_header_h

        label = METHOD_LABELS.get(method, method)
        is_bold = method in bold_methods
        font = method_font_bold if is_bold else method_font

        fitted_text, fitted_font, _ = v6.fit_prompt_to_box(
            label,
            max_width=args.thumb - 8,
            max_height=args.method_header_h - 8,
            max_font_size=args.method_font_size,
            min_font_size=args.method_min_font_size,
            serif=True,
            bold=is_bold,
            spacing=args.caption_spacing,
        )

        v6.aligned_multiline(
            draw,
            (x0, y0, x1, y1),
            fitted_text,
            fitted_font,
            spacing=args.caption_spacing,
            valign="center",
            pad_y=0,
        )

    report = []

    # Draw prompt labels and images.
    for ri, row in cases.iterrows():
        y0 = y_grid0 + ri * (args.thumb + args.row_gap)
        y1 = y0 + args.thumb

        prompt_id = str(row["prompt_id"])
        seed = int(row["seed"]) if "seed" in row.index and not pd.isna(row["seed"]) else 1
        display_prompt = get_display_prompt(row, args.prompt_text_col)
        lookup_prompt = get_lookup_prompt(row)

        # Left prompt label.
        fitted_prompt, fitted_prompt_font, _ = v6.fit_prompt_to_box(
            display_prompt,
            max_width=args.prompt_label_w - 12,
            max_height=args.thumb - 12,
            max_font_size=args.prompt_font_size,
            min_font_size=args.prompt_min_font_size,
            serif=True,
            bold=False,
            spacing=args.caption_spacing,
        )

        v6.aligned_multiline(
            draw,
            (args.outer, y0, args.outer + args.prompt_label_w, y1),
            fitted_prompt,
            fitted_prompt_font,
            spacing=args.caption_spacing,
            valign="center",
            pad_y=0,
        )

        # Method images.
        for ci, method in enumerate(methods):
            x0 = x_grid0 + ci * (args.thumb + args.col_gap)

            path, status = v6.select_image(
                index_df,
                prompt_id,
                lookup_prompt,
                method,
                seed,
            )

            img, quality, lum_mean, lum_std = v6.load_img(path, (args.thumb, args.thumb))
            canvas.paste(img, (x0, y0))

            report.append({
                "figure": args.figure,
                "prompt_id": prompt_id,
                "display_prompt": display_prompt,
                "full_prompt": lookup_prompt,
                "seed": seed,
                "method": method,
                "status": status,
                "image_quality": quality,
                "luminance_mean": lum_mean,
                "luminance_std": lum_std,
                "image_path": path,
            })

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, dpi=(args.dpi, args.dpi))

    if args.out_pdf:
        v6.save_pdf_from_rgb(canvas, args.out_pdf, dpi=args.dpi)

    report_df = pd.DataFrame(report)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.report_csv, index=False)

    print("Saved PNG:", out_png)
    if args.out_pdf:
        print("Saved PDF:", args.out_pdf)
    print("Saved report:", args.report_csv)
    print("Canvas size:", W, "x", H)
    print("status:")
    print(report_df["status"].value_counts().to_string())
    print("quality:")
    print(report_df["image_quality"].value_counts().to_string())

    missing = report_df[report_df["status"].eq("missing")]
    if len(missing):
        print("\nMissing:")
        print(missing[["prompt_id", "display_prompt", "full_prompt", "method", "seed"]].to_string(index=False))
        if args.strict:
            raise SystemExit("Strict mode: missing images detected.")

    near_black = report_df[report_df["image_quality"].eq("near_black")]
    if len(near_black):
        print("\nNear-black / failed images:")
        print(near_black[["prompt_id", "display_prompt", "method", "seed", "luminance_mean", "luminance_std", "image_path"]].to_string(index=False))
        if args.reject_near_black:
            raise SystemExit("Near-black images detected.")


if __name__ == "__main__":
    main()
