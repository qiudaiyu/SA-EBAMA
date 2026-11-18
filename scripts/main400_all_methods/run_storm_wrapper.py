#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import spacy

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from formal_utils import (
    read_prompt_csv,
    prompt_dir_name,
    external_sample_complete,
    write_json,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 1234, 70690])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_iter_to_alter", type=int, default=25)
    parser.add_argument("--attention_res", type=int, default=16)
    parser.add_argument("--scale_factor", type=int, default=20)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_error_metadata", action="store_true")
    return parser.parse_args()

def normalize_word(x):
    x = str(x).lower().strip()
    x = x.replace("</w>", "")
    x = re.sub(r"[^a-z0-9\- ]+", "", x)
    return x.strip()

def parse_json_list(x):
    try:
        v = json.loads(x)
        if isinstance(v, list):
            return [str(a) for a in v]
    except Exception:
        pass
    return []

def unique_indices(indices):
    if not indices:
        return []
    indices = sorted(set(indices))
    return [
        current for current, next_val in zip(indices, indices[1:] + [None])
        if next_val is None or current + 1 != next_val
    ]

def find_token_index_for_word(token_map, target_word, after_idx=-1):
    target_word = normalize_word(target_word)
    if not target_word:
        return None

    for idx, word in token_map.items():
        w = normalize_word(word)
        if idx <= after_idx:
            continue
        if w == target_word:
            return idx

    for idx, word in token_map.items():
        w = normalize_word(word)
        if idx <= after_idx:
            continue
        if target_word in w or w in target_word:
            return idx

    return None

def get_last_noun_and_adj(phrase, nlp):
    doc = nlp(phrase)

    nouns = [
        token for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and normalize_word(token.text) not in ["left", "right"]
    ]

    if nouns:
        noun = nouns[-1]
    else:
        words = [
            w for w in re.findall(r"[a-zA-Z]+", phrase)
            if w.lower() not in ["a", "an", "the", "of", "to"]
        ]
        noun = words[-1] if words else None

    adjs = [token for token in doc if token.pos_ == "ADJ"]

    adj = None
    if nouns:
        noun_pos = nouns[-1].i
        prev_adjs = [a for a in adjs if a.i < noun_pos]
        if prev_adjs:
            adj = prev_adjs[-1]
    elif adjs:
        adj = adjs[-1]

    noun_text = noun.text if hasattr(noun, "text") else noun
    adj_text = adj.text if hasattr(adj, "text") else adj

    return normalize_word(noun_text), normalize_word(adj_text) if adj_text else None

def fallback_prompt_noun_adj_indices(prompt, token_map, nlp):
    doc = nlp(prompt)

    noun_words = []
    adj_words = []

    for token in doc:
        txt = normalize_word(token.text)
        lemma = normalize_word(token.lemma_)

        if token.pos_ in ["NOUN", "PROPN"] and txt not in ["left", "right"]:
            noun_words.append(txt)
            if lemma and lemma != txt:
                noun_words.append(lemma)

        if token.pos_ == "ADJ":
            adj_words.append(txt)
            if lemma and lemma != txt:
                adj_words.append(lemma)

    noun_indices = []
    last = -1

    for n in noun_words:
        idx = find_token_index_for_word(token_map, n, after_idx=last)
        if idx is not None:
            noun_indices.append(idx)
            last = idx
        if len(noun_indices) >= 2:
            break

    adj_indices = [None] * len(noun_indices)

    for a in adj_words:
        aidx = find_token_index_for_word(token_map, a, after_idx=-1)
        if aidx is None:
            continue
        for k, nidx in enumerate(noun_indices):
            if aidx < nidx and adj_indices[k] is None:
                adj_indices[k] = aidx
                break

    return noun_indices[:2], adj_indices[:2]

def build_storm_token_indices(stable, prompt, subprompts_json, nlp):
    tokens = stable.tokenizer(prompt)["input_ids"]
    token_map = {
        idx: stable.tokenizer.decode(t).strip()
        for idx, t in enumerate(tokens)
        if 0 < idx < len(tokens) - 1
    }

    subprompts = parse_json_list(subprompts_json)

    noun_indices = []
    adj_indices = []

    if len(subprompts) >= 2:
        last = -1
        for phrase in subprompts[:2]:
            noun_word, adj_word = get_last_noun_and_adj(phrase, nlp)

            nidx = find_token_index_for_word(token_map, noun_word, after_idx=last)
            if nidx is None:
                nidx = find_token_index_for_word(token_map, noun_word, after_idx=-1)

            aidx = None
            if adj_word is not None:
                aidx = find_token_index_for_word(token_map, adj_word, after_idx=-1)

            if nidx is not None:
                noun_indices.append(nidx)
                adj_indices.append(aidx)
                last = nidx

    if len(noun_indices) < 2:
        noun_indices, adj_indices = fallback_prompt_noun_adj_indices(prompt, token_map, nlp)

    noun_indices = unique_indices(noun_indices)[:2]
    adj_indices = (adj_indices + [None, None])[:2]

    fallback_reason = ""

    # STORM is a two-object spatial-attention method.
    # For prompts with fewer than two detectable object tokens, fall back to standard SD generation.
    if len(noun_indices) < 2:
        usable = []
        stop_words = {
            "a", "an", "the", "with", "and", "or", "of", "to", "on", "in",
            "left", "right", "above", "below", "under", "over", "front",
            "behind", "next", "near", "beside", "side", ",", ".", ""
        }

        for idx, word in token_map.items():
            w = normalize_word(word)
            if not w or w in stop_words:
                continue
            if not re.search(r"[a-z0-9]", w):
                continue
            usable.append(idx)

        if len(noun_indices) == 1:
            noun_indices = [noun_indices[0], noun_indices[0]]
            fallback_reason = "single_object_prompt"
        elif len(usable) >= 1:
            noun_indices = [usable[0], usable[0]]
            fallback_reason = "no_object_pair_detected"
        else:
            noun_indices = [1, 1]
            fallback_reason = "no_usable_token_detected"

        adj_indices = [None, None]

    noun_indices = noun_indices[:2]
    adj_indices = (adj_indices + [None, None])[:2]

    return [noun_indices, adj_indices], token_map, subprompts, fallback_reason

def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from config import RunConfig
    from run import load_pipeline_from_config, run_on_prompt
    from utils.ptp_utils import AttentionStore

    config = RunConfig()
    config.n_inference_steps = args.num_inference_steps
    config.guidance_scale = args.guidance_scale
    config.max_iter_to_alter = args.max_iter_to_alter
    config.attention_res = args.attention_res
    config.scale_factor = args.scale_factor
    config.run_standard_sd = False
    config.sd_2_1 = False

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading STORM pipeline from:", repo_root)
    StormPipeline = load_pipeline_from_config(config)

    load_kwargs = {}
    if args.fp16:
        load_kwargs["torch_dtype"] = torch.float16

    stable = StormPipeline.from_pretrained(args.model_path, **load_kwargs).to(device)

    # Diffusers Pipeline is not a torch.nn.Module and may not have .eval().
    # Set eval mode for its neural-network components instead.
    for module_name in ["unet", "vae", "text_encoder", "safety_checker"]:
        module = getattr(stable, module_name, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()

    try:
        stable.set_progress_bar_config(disable=False)
    except Exception:
        pass

    nlp = spacy.load("en_core_web_sm")

    records = read_prompt_csv(Path(args.csv))
    records = records[args.start_idx:args.end_idx]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Total prompts in this slice: {len(records)}")
    print(f"Seeds: {args.seeds}")

    for local_idx, row in enumerate(records):
        global_idx = args.start_idx + local_idx

        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        subprompts_json = row.get("subprompts_json", "")

        prompt_dir = out_root / prompt_dir_name(prompt_id, prompt)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            image_path = prompt_dir / f"seed{seed}.png"
            meta_path = prompt_dir / f"seed{seed}_metadata.json"

            if args.resume and external_sample_complete(prompt_dir, seed) and not args.overwrite:
                print(f"[SKIP] {prompt_id} seed={seed}")
                continue

            print("=" * 100)
            print(f"[STORM] idx={global_idx} prompt_id={prompt_id} seed={seed}")
            print(prompt)

            t0 = time.time()
            error = ""
            peak_memory_mb = None
            token_indices = None
            token_map = None
            subprompts = None
            fallback_reason = ""
            sample_run_standard_sd = False

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                token_indices, token_map, subprompts, fallback_reason = build_storm_token_indices(
                    stable=stable,
                    prompt=prompt,
                    subprompts_json=subprompts_json,
                    nlp=nlp,
                )

                sample_run_standard_sd = bool(fallback_reason)

                print("subprompts:", subprompts)
                print("token_indices:", token_indices)
                print("token_map:", token_map)
                print("fallback_reason:", fallback_reason)

                generator_device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(generator_device).manual_seed(seed)

                controller = AttentionStore()

                old_run_standard_sd = config.run_standard_sd
                config.run_standard_sd = sample_run_standard_sd

                image = run_on_prompt(
                    prompt=prompt,
                    model=stable,
                    controller=controller,
                    token_indices=token_indices,
                    seed=generator,
                    config=config,
                )

                config.run_standard_sd = old_run_standard_sd

                image.save(image_path)

                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            except Exception as e:
                error = str(e)
                traceback.print_exc()

            time_seconds = time.time() - t0

            metadata = {
                "method": "storm",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "subprompts_json": subprompts_json,
                "subprompts": subprompts,
                "seed": seed,
                "image_path": str(image_path),
                "time_seconds": time_seconds,
                "peak_memory_mb": peak_memory_mb,
                "token_indices": token_indices,
                "token_map": token_map,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "max_iter_to_alter": args.max_iter_to_alter,
                "attention_res": args.attention_res,
                "scale_factor": args.scale_factor,
                "fp16": bool(args.fp16),
                "storm_fallback_standard_sd": bool(sample_run_standard_sd),
                "fallback_reason": fallback_reason,
                "error": error,
            }

            if error:
                if args.save_error_metadata:
                    write_json(meta_path, metadata)
                print(f"[ERROR] {prompt_id} seed={seed}: {error}")
            else:
                write_json(meta_path, metadata)
                print(f"[DONE] {image_path} | time={time_seconds:.2f}s")

    print("STORM generation finished.")

if __name__ == "__main__":
    main()
