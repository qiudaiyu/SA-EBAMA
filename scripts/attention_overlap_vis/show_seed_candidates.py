import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def resolve_path(project_root: Path, p: str) -> Path:
    p = Path(str(p).strip())
    if not p.is_absolute():
        p = project_root / p
    return p


def safe_name(text: str, max_len: int = 140) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {" ", "-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip().replace("  ", " ")
    return name[:max_len] if name else "prompt"


def load_image(path: Path, size: int = 256):
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--prompt_id", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--methods", nargs="+", default=["ebama", "sa_ebama"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    project_root = Path.cwd()
    index_csv = Path(args.index_csv)

    if not index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {index_csv}")

    df = pd.read_csv(index_csv)

    required_cols = {"method", "prompt_id", "prompt", "seed", "image_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in index csv: {missing}")

    df["method_norm"] = df["method"].astype(str).str.strip().str.lower()
    target_methods = [m.strip().lower() for m in args.methods]

    if args.prompt_id:
        sub = df[df["prompt_id"].astype(str).str.strip() == str(args.prompt_id).strip()].copy()
    elif args.prompt:
        prompt = str(args.prompt).strip()
        sub = df[df["prompt"].astype(str).str.strip() == prompt].copy()
        if sub.empty:
            sub = df[df["prompt"].astype(str).str.contains(prompt, case=False, regex=False)].copy()
    else:
        raise ValueError("Please provide --prompt_id or --prompt.")

    sub = sub[sub["method_norm"].isin(target_methods)].copy()

    if "exists" in sub.columns:
        sub = sub[sub["exists"].astype(int) == 1].copy()

    if sub.empty:
        raise SystemExit("No candidate images found. Check prompt_id / prompt / methods.")

    sub["abs_image_path"] = sub["image_path"].apply(lambda p: resolve_path(project_root, p))
    sub = sub[sub["abs_image_path"].apply(lambda p: p.exists())].copy()

    if sub.empty:
        raise SystemExit("Candidate rows found, but image files do not exist. Check image_path in all_images_index.csv.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = sub.sort_values(["method_norm", "seed"])
    sub.to_csv(out_dir / "seed_candidates.csv", index=False)

    methods = target_methods
    seeds = sorted(sub["seed"].astype(int).unique().tolist())

    cell = args.image_size
    label_h = 48
    left_w = 170

    canvas_w = left_w + len(seeds) * cell
    canvas_h = label_h + len(methods) * (cell + label_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for c, seed in enumerate(seeds):
        x = left_w + c * cell
        draw.text((x + 10, 12), f"seed {seed}", fill="black")

    for r, method in enumerate(methods):
        y = label_h + r * (cell + label_h)
        draw.text((10, y + cell // 2 - 10), method, fill="black")

        for c, seed in enumerate(seeds):
            x = left_w + c * cell
            row = sub[
                (sub["method_norm"] == method)
                & (sub["seed"].astype(int) == int(seed))
            ]

            if row.empty:
                draw.rectangle([x, y, x + cell, y + cell], outline="gray")
                draw.text((x + 10, y + 10), "missing", fill="red")
                continue

            image_path = row.iloc[0]["abs_image_path"]

            try:
                img = load_image(image_path, size=cell)
                canvas.paste(img, (x, y))
                draw.rectangle([x, y, x + cell, y + cell], outline="black", width=1)
            except Exception as e:
                draw.rectangle([x, y, x + cell, y + cell], outline="red")
                draw.text((x + 10, y + 10), f"load error\n{e}", fill="red")

    prompt_text = sub.iloc[0]["prompt"]
    prompt_id = sub.iloc[0]["prompt_id"]

    out_name = safe_name(f"{prompt_id}_{prompt_text}")
    out_path = out_dir / f"{out_name}_seed_candidates.png"
    canvas.save(out_path)

    print("Saved:")
    print(out_dir / "seed_candidates.csv")
    print(out_path)


if __name__ == "__main__":
    main()
