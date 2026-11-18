#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from formal_utils import (
    read_prompt_csv,
    prompt_dir_name,
    ensure_dir,
    write_json,
    image_complete,
)

SEEDS_DEFAULT = [1, 42, 1234, 70690]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--model_path", required=True)

    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--scheduler", type=str, default="ddim")

    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    # CONFORM official-style parameters
    p.add_argument("--max_iter_to_alter", type=int, default=25)
    p.add_argument("--refinement_steps", type=int, default=20)
    p.add_argument("--scale_factor", type=float, default=20.0)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--loss_fn", type=str, default="ntxent")

    # Practical option for large-scale evaluation if official-style setting is too slow
    p.add_argument("--practical", action="store_true")

    # For prompts with only one semantic group, CONFORM contrastive loss is not applicable.
    # sd_fallback keeps the sample in the full benchmark by disabling CONFORM latent update.
    p.add_argument(
        "--single_group_policy",
        choices=["sd_fallback", "skip"],
        default="sd_fallback",
    )

    return p.parse_args()


def parse_subprompts(raw):
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    return []


def split_prompt_fallback(prompt):
    relations = [
        " to the left of ",
        " to the right of ",
        " on the left of ",
        " on the right of ",
        " in front of ",
        " next to ",
        " beside ",
        " behind ",
        " above ",
        " below ",
        " under ",
        " over ",
        " near ",
        " with ",
        " and ",
    ]
    for sep in relations:
        if sep in prompt:
            parts = [p.strip() for p in prompt.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts
    return [prompt]


def phrase_content_words(phrase):
    stop = {
        "a", "an", "the", "and", "with", "on", "in", "at", "to", "of",
        "left", "right", "front", "behind", "above", "below", "under",
        "over", "near", "next", "beside",
    }
    words = []
    for w in str(phrase).replace("-", " ").split():
        w = w.strip(" ,.;:!?()[]{}\"'").lower()
        if w and w not in stop:
            words.append(w)
    return words


def clean_token(tok):
    tok = str(tok).lower()
    tok = tok.replace("</w>", "")
    tok = tok.replace("Ġ", "")
    tok = tok.replace("▁", "")
    tok = tok.strip()
    return tok


def find_word_indices(tokenizer, prompt, target_word, used=None):
    if used is None:
        used = set()

    ids = tokenizer(prompt, add_special_tokens=True).input_ids
    toks = tokenizer.convert_ids_to_tokens(ids)
    clean = [clean_token(t) for t in toks]

    target_word = target_word.lower().strip()
    found = []

    for i, t in enumerate(clean):
        if i in used or not t:
            continue
        if t == target_word:
            found.append(i)
        elif len(target_word) > 3 and (t.endswith(target_word) or target_word.endswith(t)):
            found.append(i)
        elif len(target_word) > 3 and target_word in t:
            found.append(i)

    if found:
        return found

    try:
        target_ids = tokenizer(target_word, add_special_tokens=False).input_ids
        target_toks = [clean_token(t) for t in tokenizer.convert_ids_to_tokens(target_ids)]
        target_toks = [t for t in target_toks if t]
        n = len(target_toks)

        if n > 0:
            for i in range(0, len(clean) - n + 1):
                if any(j in used for j in range(i, i + n)):
                    continue
                if clean[i:i+n] == target_toks:
                    return list(range(i, i+n))
    except Exception:
        pass

    return []


def infer_single_token_group(tokenizer, prompt):
    """
    Infer one semantic group for single-object prompts.
    This is used only for SD fallback mode with CONFORM latent update disabled.
    """
    words = phrase_content_words(prompt)
    used = set()
    group = []

    for w in words:
        idxs = find_word_indices(tokenizer, prompt, w, used=used)
        for idx in idxs:
            if idx > 0 and idx not in group:
                group.append(idx)
                used.add(idx)

    if not group:
        # Minimal fallback for prompts like "a tomato".
        ids = tokenizer(prompt, add_special_tokens=True).input_ids
        toks = tokenizer.convert_ids_to_tokens(ids)
        # Use all non-special tokens except articles if possible.
        for i, tok in enumerate(toks):
            clean = clean_token(tok)
            if i > 0 and clean not in {"a", "an", "the", "", "<|endoftext|>"}:
                group.append(i)

    if not group:
        raise ValueError(f"Cannot infer single token group for prompt={prompt!r}")

    return [sorted(set(group))]


def infer_token_groups(tokenizer, prompt, subprompts_json):
    """
    CONFORM needs token_groups: List[List[int]].

    Example:
    "a monkey and a turtle" -> [[2], [5]]
    "a green glasses and a yellow clock" -> [[2, 3], [6, 7]]
    """
    subprompts = parse_subprompts(subprompts_json)
    if len(subprompts) < 2:
        subprompts = split_prompt_fallback(prompt)

    groups = []
    used = set()

    for sp in subprompts:
        words = phrase_content_words(sp)
        group = []
        for w in words:
            idxs = find_word_indices(tokenizer, prompt, w, used=used)
            for idx in idxs:
                if idx > 0 and idx not in group:
                    group.append(idx)
                    used.add(idx)
        if group:
            groups.append(sorted(set(group)))

    # Fallback for simple A&E-style prompts.
    if len(groups) < 2:
        words = prompt.split()
        if len(words) == 5:
            groups = [[2], [5]]
        elif len(words) == 6:
            groups = [[2], [5, 6]]
        elif len(words) == 7:
            groups = [[2, 3], [6, 7]]

    if len(groups) < 2:
        raise ValueError(
            f"Cannot infer token_groups for prompt={prompt!r}, subprompts={subprompts!r}"
        )

    return groups


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from pipeline_conform import ConformPipeline

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    records = read_prompt_csv(Path(args.csv))
    if args.end_idx is None:
        records = records[args.start_idx:]
    else:
        records = records[args.start_idx:args.end_idx]

    out_root = ensure_dir(Path(args.out_root))

    print("Loading CONFORM pipeline")
    print("repo_root:", repo_root)
    print("model_path:", args.model_path)
    print("device:", device)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Use StableDiffusionPipeline first, then build ConformPipeline from its components.
    # This follows the official notebook / code style and works with local SD1.4 path.
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    if str(args.scheduler).lower() == "ddim":
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)

    pipe = ConformPipeline(
        vae=sd_pipe.vae,
        text_encoder=sd_pipe.text_encoder,
        tokenizer=sd_pipe.tokenizer,
        unet=sd_pipe.unet,
        scheduler=sd_pipe.scheduler,
        safety_checker=None,
        feature_extractor=sd_pipe.feature_extractor,
        requires_safety_checker=False,
    ).to(device)

    attn_res = (16, 16)

    max_iter_to_alter = int(args.max_iter_to_alter)
    refinement_steps = int(args.refinement_steps)

    if args.practical:
        max_iter_to_alter = min(max_iter_to_alter, 10)
        refinement_steps = min(refinement_steps, 10)

    for row in records:
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        subprompts_json = row.get("subprompts_json", "")

        prompt_dir = ensure_dir(out_root / prompt_dir_name(prompt_id, prompt))

        single_group_fallback = False

        try:
            token_groups = infer_token_groups(pipe.tokenizer, prompt, subprompts_json)
        except Exception as e:
            if args.single_group_policy == "sd_fallback":
                try:
                    token_groups = infer_single_token_group(pipe.tokenizer, prompt)
                    single_group_fallback = True
                    write_json(prompt_dir / "token_group_fallback.json", {
                        "method": "conform",
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "subprompts_json": subprompts_json,
                        "token_groups": token_groups,
                "single_group_fallback": bool(single_group_fallback),
                "single_group_policy": args.single_group_policy,
                        "policy": "sd_fallback",
                        "reason": repr(e),
                    })
                    print("[TOKEN_GROUP_FALLBACK]", prompt_id, prompt, token_groups, repr(e))
                except Exception as e2:
                    write_json(prompt_dir / "token_group_error.json", {
                        "method": "conform",
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "subprompts_json": subprompts_json,
                        "error": repr(e2),
                        "original_error": repr(e),
                    })
                    print("[TOKEN_GROUP_ERROR]", prompt_id, prompt, repr(e2))
                    continue
            else:
                write_json(prompt_dir / "token_group_error.json", {
                    "method": "conform",
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "subprompts_json": subprompts_json,
                    "error": repr(e),
                })
                print("[TOKEN_GROUP_ERROR]", prompt_id, prompt, repr(e))
                continue

        print("=" * 100)
        print("prompt_id:", prompt_id)
        print("prompt:", prompt)
        print("token_groups:", token_groups)

        for seed in args.seeds:
            print(f"[START] prompt_id={prompt_id} seed={seed}", flush=True)

            image_path = prompt_dir / f"seed{seed}.png"
            meta_path = prompt_dir / f"seed{seed}_metadata.json"

            if args.resume and image_complete(image_path) and meta_path.exists() and not args.overwrite:
                print("[SKIP]", image_path)
                continue

            set_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator("cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

            start = time.perf_counter()
            error = ""

            try:
                if single_group_fallback:
                    sample_max_iter_to_alter = 0
                    sample_refinement_steps = 0
                    sample_iterative_refinement_steps = []
                else:
                    sample_max_iter_to_alter = max_iter_to_alter
                    sample_refinement_steps = refinement_steps
                    sample_iterative_refinement_steps = [0, 10, 20]

                images, attention_maps = pipe(
                    prompt=prompt,
                    token_groups=token_groups,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    max_iter_to_alter=sample_max_iter_to_alter,
                    refinement_steps=sample_refinement_steps,
                    iterative_refinement_steps=sample_iterative_refinement_steps,
                    scale_factor=args.scale_factor,
                    attn_res=attn_res,
                    steps_to_save_attention_maps=None,
                    do_smoothing=True,
                    smoothing_sigma=0.5,
                    smoothing_kernel_size=3,
                    temperature=args.temperature,
                    softmax_normalize=False,
                    softmax_normalize_attention_maps=False,
                    add_previous_attention_maps=True,
                    previous_attention_map_anchor_step=None,
                    loss_fn=args.loss_fn,
                )

                images[0].save(image_path)

            except Exception:
                error = traceback.format_exc()
                print("[ERROR]", prompt_id, seed, error)

            elapsed = time.perf_counter() - start

            peak_memory_mb = None
            if torch.cuda.is_available():
                peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)

            write_json(meta_path, {
                "method": "conform",
                "variant": "conform_practical" if args.practical else "conform",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "subprompts_json": subprompts_json,
                "seed": int(seed),
                "image_path": str(image_path),
                "time_seconds": float(elapsed),
                "peak_memory_mb": peak_memory_mb,
                "token_groups": token_groups,
                "single_group_fallback": bool(single_group_fallback),
                "single_group_policy": args.single_group_policy,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "scheduler": args.scheduler,
                "attn_res": attn_res,
                "max_iter_to_alter": sample_max_iter_to_alter if 'sample_max_iter_to_alter' in locals() else max_iter_to_alter,
                "refinement_steps": sample_refinement_steps if 'sample_refinement_steps' in locals() else refinement_steps,
                "scale_factor": args.scale_factor,
                "temperature": args.temperature,
                "loss_fn": args.loss_fn,
                "practical": bool(args.practical),
                "model_path": args.model_path,
                "repo_root": str(repo_root),
                "error": error,
            })

            print("[DONE]", prompt_id, "seed", seed, "time", elapsed, "error", error)

    print("CONFORM formal wrapper finished.")


if __name__ == "__main__":
    main()
