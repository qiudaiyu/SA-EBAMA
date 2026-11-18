

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rgb(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_heatmap(vis_dir):
    vis_dir = Path(vis_dir)
    heatmap_path = vis_dir / "avg_overlap_heatmap.npy"
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")
    return np.load(heatmap_path)


def read_oor(vis_dir):
    vis_dir = Path(vis_dir)
    meta_path = vis_dir / "meta.json"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in ["avg_oor", "mean_oor", "oor", "avg_overlap", "mean_overlap"]:
                if key in meta:
                    return float(meta[key])
        except Exception:
            pass


    heatmap_path = vis_dir / "avg_overlap_heatmap.npy"
    if heatmap_path.exists():
        h = np.load(heatmap_path)
        return float(np.nanmean(h))

    return float("nan")


def wrap_text(text, width=70):
    text = str(text)
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--ebama_image", required=True)
    parser.add_argument("--sa_image", required=True)
    parser.add_argument("--ebama_vis_dir", required=True)
    parser.add_argument("--sa_vis_dir", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_pdf", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    ebama_img = load_rgb(args.ebama_image)
    sa_img = load_rgb(args.sa_image)

    ebama_heat = load_heatmap(args.ebama_vis_dir)
    sa_heat = load_heatmap(args.sa_vis_dir)

    ebama_oor = read_oor(args.ebama_vis_dir)
    sa_oor = read_oor(args.sa_vis_dir)
    delta_oor = ebama_oor - sa_oor


    vmax = float(max(np.nanmax(ebama_heat), np.nanmax(sa_heat), 1e-8))


    reduced = np.maximum(ebama_heat - sa_heat, 0.0)
    reduced_vmax = float(max(np.nanmax(reduced), 1e-8))

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))

    axes[0].imshow(ebama_img)
    axes[0].set_title("EBAMA Image", fontsize=15)

    axes[1].imshow(
        ebama_heat,
        cmap="jet",
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].set_title(f"EBAMA Overlap\nOOR={ebama_oor:.4f}", fontsize=15)

    axes[2].imshow(sa_img)
    axes[2].set_title("SA-EBAMA Image", fontsize=15)

    axes[3].imshow(
        sa_heat,
        cmap="jet",
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax,
    )
    axes[3].set_title(f"SA-EBAMA Overlap\nOOR={sa_oor:.4f}", fontsize=15)

    axes[4].imshow(
        reduced,
        cmap="jet",
        interpolation="bilinear",
        vmin=0.0,
        vmax=reduced_vmax,
    )
    axes[4].set_title(f"Reduced Overlap\nΔOOR={delta_oor:.4f}", fontsize=15)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Prompt: " + wrap_text(args.prompt, width=90),
        fontsize=16,
        y=1.08,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)

    plt.close()

    print("Saved:", out_png)
    if args.out_pdf:
        print("Saved:", args.out_pdf)


if __name__ == "__main__":
    main()
