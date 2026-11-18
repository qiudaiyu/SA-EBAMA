import argparse
import re
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# v6: paper-ready qualitative grid builder with pixel-aware prompt fitting.
# Relative to v3/v5:
# 1) Supports display-only prompt labels via --prompt_text_col or automatic columns:
#    display_prompt / short_prompt / prompt_label.
# 2) Keeps original prompts for image lookup and report via full_prompt when available.
# 3) Supports bold prompt labels, vertical alignment, and automatic prompt-area height.
# 4) Retains CONFORM support, PDF export, strict missing-image checks, and near-black checks.

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


def norm_text(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def parse_seed_from_path(p):
    s = str(p)
    m = re.search(r"seed(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(^|[_/])(\d+)\.(png|jpg|jpeg|webp)$", s, flags=re.I)
    if m:
        return int(m.group(2))
    return None


def load_font(size=24, serif=True, bold=False):
    font_paths = []
    if serif:
        if bold:
            font_paths += [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            ]
        font_paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        ]

    if bold:
        font_paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]

    font_paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for p in font_paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def resolve_path(project_root, p):
    p = Path(str(p))
    if p.is_absolute() and p.exists():
        return p
    p2 = Path(project_root) / p
    if p2.exists():
        return p2
    return p


def load_index(index_csv, project_root):
    df = pd.read_csv(index_csv)

    method_col = find_col(df, ["method", "method_name", "model", "approach"])
    prompt_id_col = find_col(df, ["prompt_id", "id"])
    prompt_col = find_col(df, ["prompt", "text", "caption"])
    seed_col = find_col(df, ["seed", "random_seed"])
    path_col = find_col(df, ["image_path", "path", "file_path", "filepath", "image", "png_path"])

    missing = []
    for name, col in [
        ("method", method_col),
        ("prompt_id", prompt_id_col),
        ("prompt", prompt_col),
        ("image_path", path_col),
    ]:
        if col is None:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing columns {missing}. Existing columns: {df.columns.tolist()}")

    df = df.copy()
    df["_method"] = df[method_col].map(norm_method)
    df["_prompt_id"] = df[prompt_id_col].astype(str)
    df["_prompt"] = df[prompt_col].astype(str)
    df["_prompt_norm"] = df["_prompt"].map(norm_text)
    df["_seed"] = df[seed_col].astype(int) if seed_col else df[path_col].map(parse_seed_from_path)
    df["_path"] = df[path_col].map(lambda x: str(resolve_path(project_root, x)))
    return df


def select_image(index_df, prompt_id, prompt, method, seed):
    method = norm_method(method)
    prompt_id = str(prompt_id)
    prompt_norm = norm_text(prompt)
    seed = int(seed)

    q = index_df[
        index_df["_prompt_id"].eq(prompt_id)
        & index_df["_method"].eq(method)
        & index_df["_seed"].eq(seed)
    ]
    if len(q):
        return q.iloc[0]["_path"], "prompt_id_seed"

    q = index_df[
        index_df["_prompt_norm"].eq(prompt_norm)
        & index_df["_method"].eq(method)
        & index_df["_seed"].eq(seed)
    ]
    if len(q):
        return q.iloc[0]["_path"], "prompt_text_seed"

    q = index_df[
        index_df["_prompt_id"].eq(prompt_id)
        & index_df["_method"].eq(method)
    ].sort_values("_seed")
    if len(q):
        return q.iloc[0]["_path"], "prompt_id_any_seed"

    q = index_df[
        index_df["_prompt_norm"].eq(prompt_norm)
        & index_df["_method"].eq(method)
    ].sort_values("_seed")
    if len(q):
        return q.iloc[0]["_path"], "prompt_text_any_seed"

    return None, "missing"


def is_near_black(im, mean_thr=8.0, std_thr=4.0):
    small = im.convert("L").resize((32, 32), Image.BILINEAR)
    pix = list(small.getdata())
    mean = sum(pix) / len(pix)
    var = sum((x - mean) ** 2 for x in pix) / len(pix)
    std = var ** 0.5
    return mean < mean_thr and std < std_thr, mean, std


def load_img(path, size):
    if path is None or not Path(path).exists():
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((15, 15), "Missing", fill=(180, 0, 0), font=load_font(18, serif=False))
        return img, "missing_file", None, None

    im0 = Image.open(path).convert("RGB")
    near_black, lum_mean, lum_std = is_near_black(im0)

    im = im0.copy()
    im.thumbnail(size, Image.LANCZOS)

    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    quality = "near_black" if near_black else "ok"
    return canvas, quality, lum_mean, lum_std


def draw_rotated_text(base, box, text, fnt):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    tmp = Image.new("RGBA", (h, w), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=fnt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((h - tw) // 2, (w - th) // 2), text, fill=(0, 0, 0, 255), font=fnt)

    rot = tmp.rotate(90, expand=True)
    base.paste(rot, (x0 + (w - rot.width) // 2, y0 + (h - rot.height) // 2), rot)


def wrap_prompt(prompt, width):
    # Respect manual line breaks first, then wrap each line separately.
    parts = str(prompt).split("\n")
    wrapped = []
    for part in parts:
        if part.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(part, width=width))
    return "\n".join(wrapped)


def text_width(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return max(0, bb[2] - bb[0])


def wrap_prompt_pixels(prompt, font, max_width):
    """Wrap text by rendered pixel width, not character count.

    This prevents adjacent prompt captions from overlapping when the font is
    large or bold. Manual line breaks are preserved.
    """
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    all_lines = []

    for part in str(prompt).split("\n"):
        part = part.strip()
        if not part:
            all_lines.append("")
            continue

        words = part.split()
        line = ""
        for word in words:
            cand = word if line == "" else line + " " + word
            if text_width(draw, cand, font) <= max_width:
                line = cand
            else:
                if line:
                    all_lines.append(line)
                # If a single token is wider than max_width, split it at char level.
                if text_width(draw, word, font) <= max_width:
                    line = word
                else:
                    piece = ""
                    for ch in word:
                        cand2 = piece + ch
                        if text_width(draw, cand2, font) <= max_width:
                            piece = cand2
                        else:
                            if piece:
                                all_lines.append(piece)
                            piece = ch
                    line = piece
        if line:
            all_lines.append(line)

    return "\n".join(all_lines)


def fit_prompt_to_box(prompt, max_width, max_height, max_font_size, min_font_size,
                      serif=True, bold=False, spacing=4):
    """Return (wrapped_text, font, font_size) that fits within a prompt cell."""
    min_font_size = min(min_font_size, max_font_size)
    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(size, serif=serif, bold=bold)
        wrapped = wrap_prompt_pixels(prompt, font, max_width)
        _, h, _ = measure_multiline(wrapped, font, spacing=spacing)
        if h <= max_height:
            return wrapped, font, size

    font = load_font(min_font_size, serif=serif, bold=bold)
    wrapped = wrap_prompt_pixels(prompt, font, max_width)
    return wrapped, font, min_font_size


def measure_multiline(text, fnt, spacing=4):
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    lines = str(text).split("\n")
    if not lines:
        return 0, 0, 0
    boxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [max(0, b[2] - b[0]) for b in boxes]
    heights = [max(0, b[3] - b[1]) for b in boxes]
    total_h = sum(heights) + spacing * max(0, len(lines) - 1)
    return max(widths) if widths else 0, total_h, len(lines)


def aligned_multiline(draw, box, text, fnt, spacing=4, valign="center", pad_y=0):
    x0, y0, x1, y1 = box
    lines = str(text).split("\n")
    boxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [max(0, b[2] - b[0]) for b in boxes]
    heights = [max(0, b[3] - b[1]) for b in boxes]
    total_h = sum(heights) + spacing * max(0, len(lines) - 1)

    if valign == "top":
        y = y0 + pad_y
    elif valign == "bottom":
        y = y1 - total_h - pad_y
    else:
        y = y0 + max(0, (y1 - y0 - total_h) // 2)

    for line, w, h in zip(lines, widths, heights):
        x = x0 + max(0, (x1 - x0 - w) // 2)
        draw.text((x, y), line, fill=(0, 0, 0), font=fnt)
        y += h + spacing


def save_pdf_from_rgb(img, out_pdf, dpi=300):
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_pdf, "PDF", resolution=float(dpi))


def get_display_prompt(row, args):
    # Image lookup should use original prompt; display can use short labels.
    if args.prompt_text_col and args.prompt_text_col in row.index and not pd.isna(row[args.prompt_text_col]):
        return str(row[args.prompt_text_col])

    for c in ["display_prompt", "short_prompt", "prompt_label", "paper_prompt"]:
        if c in row.index and not pd.isna(row[c]) and str(row[c]).strip():
            return str(row[c])

    return str(row["prompt"])


def get_lookup_prompt(row):
    if "full_prompt" in row.index and not pd.isna(row["full_prompt"]) and str(row["full_prompt"]).strip():
        return str(row["full_prompt"])
    return str(row["prompt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", default=None)
    ap.add_argument("--report_csv", required=True)
    ap.add_argument("--project_root", default=".")

    # Compact layout controls
    ap.add_argument("--thumb", type=int, default=300)
    ap.add_argument("--col_gap", type=int, default=18)
    ap.add_argument("--row_gap", type=int, default=8)
    ap.add_argument("--label_w", type=int, default=70)
    ap.add_argument("--outer", type=int, default=8)
    ap.add_argument("--prompt_gap", type=int, default=12)
    ap.add_argument("--prompt_h", type=int, default=120)

    ap.add_argument("--prompt_wrap", type=int, default=22)
    ap.add_argument("--method_font_size", type=int, default=26)
    ap.add_argument("--prompt_font_size", type=int, default=21)
    ap.add_argument("--caption_spacing", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=300)

    # Paper layout controls
    ap.add_argument(
        "--prompt_position",
        choices=["top", "bottom"],
        default="bottom",
        help="Place prompt captions above or below the image grid.",
    )
    ap.add_argument(
        "--bold_methods",
        nargs="*",
        default=["sa_ebama"],
        help="Method names to render in bold. Example: --bold_methods sa_ebama ours",
    )
    ap.add_argument(
        "--prompt_text_col",
        default=None,
        help="Optional manifest column used only for displayed prompt labels. If omitted, tries display_prompt/short_prompt/prompt_label, then prompt.",
    )
    ap.add_argument("--prompt_font_bold", action="store_true", help="Render displayed prompt labels in bold.")
    ap.add_argument(
        "--prompt_valign",
        choices=["top", "center", "bottom"],
        default="center",
        help="Vertical alignment of prompt text inside the prompt area.",
    )
    ap.add_argument("--prompt_pad_y", type=int, default=2)
    ap.add_argument("--auto_prompt_h", action="store_true", help="Automatically set prompt_h from displayed labels.")
    ap.add_argument("--prompt_h_min", type=int, default=70)
    ap.add_argument("--prompt_h_pad", type=int, default=14)

    # v6 additions: robust, pixel-aware prompt rendering.
    ap.add_argument("--prompt_fit_to_box", action="store_true",
                    help="Use pixel-aware wrapping and shrink prompt font only when necessary to avoid overlap.")
    ap.add_argument("--prompt_min_font_size", type=int, default=18,
                    help="Minimum font size allowed when --prompt_fit_to_box is enabled.")
    ap.add_argument("--prompt_inner_pad_x", type=int, default=6,
                    help="Horizontal padding inside each prompt caption box.")
    ap.add_argument("--prompt_inner_pad_y", type=int, default=4,
                    help="Vertical padding inside each prompt caption box.")
    ap.add_argument("--prompt_no_bold", action="store_true",
                    help="Convenience flag to disable bold prompt labels even if --prompt_font_bold is passed.")

    ap.add_argument("--strict", action="store_true", help="raise an error if any requested image is missing")
    ap.add_argument("--reject_near_black", action="store_true", help="raise an error if an existing image is almost all black")
    args = ap.parse_args()

    index_df = load_index(args.index_csv, args.project_root)
    manifest = pd.read_csv(args.manifest_csv)
    if "figure" in manifest.columns:
        cases = manifest[manifest["figure"].eq(args.figure)].sort_values("row_order").copy()
    else:
        cases = manifest.sort_values("row_order").copy()

    if cases.empty:
        raise ValueError(f"No rows for figure={args.figure}")

    methods = [norm_method(m) for m in args.methods]
    bold_methods = {norm_method(m) for m in args.bold_methods}
    n_methods = len(methods)
    n_cases = len(cases)
    thumb = args.thumb

    method_font = load_font(args.method_font_size, serif=True, bold=False)
    method_font_bold = load_font(args.method_font_size, serif=True, bold=True)
    prompt_bold = bool(args.prompt_font_bold) and not bool(args.prompt_no_bold)
    prompt_font = load_font(args.prompt_font_size, serif=True, bold=prompt_bold)

    display_prompts = [get_display_prompt(row, args) for _, row in cases.iterrows()]

    # Estimate prompt height before drawing.
    if args.prompt_fit_to_box:
        # Pixel-aware wrapping avoids horizontal overlap. Use the requested font
        # size first to estimate the caption-row height.
        max_w = max(20, thumb - 2 * args.prompt_inner_pad_x)
        wrapped_display_prompts = [
            wrap_prompt_pixels(x, prompt_font, max_w) for x in display_prompts
        ]
    else:
        wrapped_display_prompts = [wrap_prompt(x, args.prompt_wrap) for x in display_prompts]

    if args.auto_prompt_h:
        max_h = 0
        for txt in wrapped_display_prompts:
            _, h, _ = measure_multiline(txt, prompt_font, spacing=args.caption_spacing)
            max_h = max(max_h, h)
        args.prompt_h = max(args.prompt_h_min, int(max_h + 2 * args.prompt_h_pad))

    grid_h = n_methods * thumb + (n_methods - 1) * args.row_gap
    W = (
        args.outer
        + args.label_w
        + n_cases * thumb
        + (n_cases - 1) * args.col_gap
        + args.outer
    )
    H = (
        args.outer
        + args.prompt_h
        + args.prompt_gap
        + grid_h
        + args.outer
    )

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    report = []

    # Layout offsets.
    x_start = args.outer + args.label_w
    if args.prompt_position == "top":
        prompt_y0 = args.outer
        prompt_y1 = args.outer + args.prompt_h
        grid_y0 = args.outer + args.prompt_h + args.prompt_gap
    else:
        grid_y0 = args.outer
        prompt_y0 = args.outer + grid_h + args.prompt_gap
        prompt_y1 = H - args.outer

    # method labels
    for mi, m in enumerate(methods):
        y = grid_y0 + mi * (thumb + args.row_gap)
        fnt = method_font_bold if m in bold_methods else method_font
        draw_rotated_text(
            canvas,
            (args.outer, y, args.outer + args.label_w, y + thumb),
            METHOD_LABELS.get(m, m),
            fnt,
        )

    # images and prompts
    for ci, ((_, row), display_prompt, wrapped_prompt) in enumerate(zip(cases.iterrows(), display_prompts, wrapped_display_prompts)):
        x = x_start + ci * (thumb + args.col_gap)

        prompt_id = str(row["prompt_id"])
        lookup_prompt = get_lookup_prompt(row)
        seed = int(row["seed"]) if "seed" in row and not pd.isna(row["seed"]) else 1

        # Prompt labels. In paper mode, use pixel-aware wrapping and shrink only
        # when needed so that captions never overlap adjacent columns or images.
        if args.prompt_fit_to_box:
            max_w = max(20, thumb - 2 * args.prompt_inner_pad_x)
            max_h = max(20, args.prompt_h - 2 * args.prompt_inner_pad_y)
            fitted_text, fitted_font, fitted_size = fit_prompt_to_box(
                display_prompt,
                max_width=max_w,
                max_height=max_h,
                max_font_size=args.prompt_font_size,
                min_font_size=args.prompt_min_font_size,
                serif=True,
                bold=prompt_bold,
                spacing=args.caption_spacing,
            )
            draw_box = (
                x + args.prompt_inner_pad_x,
                prompt_y0 + args.prompt_inner_pad_y,
                x + thumb - args.prompt_inner_pad_x,
                prompt_y1 - args.prompt_inner_pad_y,
            )
            aligned_multiline(
                draw,
                draw_box,
                fitted_text,
                fitted_font,
                spacing=args.caption_spacing,
                valign=args.prompt_valign,
                pad_y=args.prompt_pad_y,
            )
        else:
            aligned_multiline(
                draw,
                (x, prompt_y0, x + thumb, prompt_y1),
                wrapped_prompt,
                prompt_font,
                spacing=args.caption_spacing,
                valign=args.prompt_valign,
                pad_y=args.prompt_pad_y,
            )

        # Images
        for mi, m in enumerate(methods):
            y = grid_y0 + mi * (thumb + args.row_gap)
            path, status = select_image(index_df, prompt_id, lookup_prompt, m, seed)
            img, quality, lum_mean, lum_std = load_img(path, (thumb, thumb))
            canvas.paste(img, (x, y))

            report.append({
                "figure": args.figure,
                "prompt_id": prompt_id,
                "display_prompt": display_prompt,
                "full_prompt": lookup_prompt,
                "seed": seed,
                "method": m,
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
        save_pdf_from_rgb(canvas, args.out_pdf, dpi=args.dpi)

    report = pd.DataFrame(report)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_csv, index=False)

    print("Saved PNG:", out_png)
    if args.out_pdf:
        print("Saved PDF:", args.out_pdf)
    print("Saved report:", args.report_csv)
    print("Canvas size:", W, "x", H)
    print("prompt_h:", args.prompt_h)
    print(report["status"].value_counts().to_string())

    missing = report[report["status"].eq("missing")]
    if len(missing):
        print("\nMissing:")
        print(missing[["prompt_id", "display_prompt", "full_prompt", "method", "seed"]].to_string(index=False))
        if args.strict:
            raise SystemExit("Strict mode: missing images detected. Check index_csv or method names.")

    near_black = report[report["image_quality"].eq("near_black")]
    if len(near_black):
        print("\nNear-black / failed images:")
        print(near_black[["prompt_id", "display_prompt", "method", "seed", "luminance_mean", "luminance_std", "image_path"]].to_string(index=False))
        if args.reject_near_black:
            raise SystemExit("Near-black images detected. Replace the case or regenerate the failed baseline image.")


if __name__ == "__main__":
    main()
