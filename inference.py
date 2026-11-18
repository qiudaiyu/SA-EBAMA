import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm


def main(
    prompts,
    seeds,
    output_directory,
    model_path,
    step_sizes,
    attn_res,
    gpu,
    dist,
    lambda_ours,
    prompt_ids=None,
    method_name="sa_ebama",
    use_separation=True,
    gamma_sep=0.10,
    sep_temp=0.15,
    sep_tau="percentile",
    sep_percentile=0.65,
    use_adaptive_guidance=True,
    adaptive_mode="sparse10",
    max_guidance_step=25,
    adaptive_loss_threshold=None,
    adaptive_overlap_threshold=0.35,
    adaptive_dynamic_step=True,
    log_guidance=True,
    force_ddim=True,

    save_overlap_vis=False,
    overlap_vis_steps="0,1,2,4,6,8,10,12,16,20",
):
    import utils.vis_utils as vis_utils

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)

    if prompt_ids is None:
        prompt_ids = [None] * len(prompts)

    if len(prompt_ids) != len(prompts):
        raise ValueError("prompt_ids must have the same length as prompts.")

    pipe = load_model(model_path, gpu, force_ddim=force_ddim)
    pipe.dist = dist
    pipe.lambda_ours = lambda_ours

    run_metadata = {
        "method": method_name,
        "model_path": model_path,
        "scheduler": pipe.scheduler.__class__.__name__,
        "dist": dist,
        "lambda_ours": lambda_ours,
        "use_separation": use_separation,
        "gamma_sep": gamma_sep,
        "sep_temp": sep_temp,
        "sep_tau": sep_tau,
        "sep_percentile": sep_percentile,
        "use_adaptive_guidance": use_adaptive_guidance,
        "adaptive_mode": adaptive_mode,
        "max_guidance_step": max_guidance_step,
        "adaptive_loss_threshold": adaptive_loss_threshold,
        "adaptive_overlap_threshold": adaptive_overlap_threshold,
        "adaptive_dynamic_step": adaptive_dynamic_step,
        "seeds": [int(s.item()) if hasattr(s, "item") else int(s) for s in seeds],
        "prompts": prompts,
        "prompt_ids": prompt_ids,
        "step_sizes": step_sizes,
        "attn_res": attn_res,
        "save_overlap_vis": save_overlap_vis,
        "overlap_vis_steps": overlap_vis_steps,
    }
    with open(output_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    for prompt, prompt_id in tqdm(list(zip(prompts, prompt_ids))):
        images = []
        safe_prompt = _safe_filename(prompt)

        if prompt_id:
            prompt_dir = output_root / f"{prompt_id}__{safe_prompt}"
        else:
            prompt_dir = output_root / safe_prompt

        prompt_dir.mkdir(parents=True, exist_ok=True)

        for step_size in tqdm(step_sizes):
            for seed in seeds:
                seed = seed.item() if hasattr(seed, "item") else int(seed)
                print(
                    f'Running on: "{prompt}" | prompt_id={prompt_id} '
                    f"| method={method_name} | seed={seed} | step_size={step_size}"
                )

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(gpu)

                suffix = f"seed{seed}_alpha{step_size:g}"
                overlap_vis_dir = None
                if save_overlap_vis:
                    overlap_vis_dir = prompt_dir / f"{suffix}_overlap_vis"

                start_time = time.perf_counter()
                image, trace = generate(
                    pipe,
                    prompt,
                    seed,
                    step_size,
                    attn_res,
                    use_separation=use_separation,
                    gamma_sep=gamma_sep,
                    sep_temp=sep_temp,
                    sep_tau=sep_tau,
                    sep_percentile=sep_percentile,
                    use_adaptive_guidance=use_adaptive_guidance,
                    adaptive_mode=adaptive_mode,
                    max_guidance_step=max_guidance_step,
                    adaptive_loss_threshold=adaptive_loss_threshold,
                    adaptive_overlap_threshold=adaptive_overlap_threshold,
                    adaptive_dynamic_step=adaptive_dynamic_step,
                    log_guidance=log_guidance,

                    save_overlap_vis=save_overlap_vis,
                    overlap_vis_dir=str(overlap_vis_dir) if overlap_vis_dir is not None else None,
                    overlap_vis_steps=overlap_vis_steps,
                    overlap_vis_method=method_name,
                    overlap_vis_prompt_id=prompt_id,
                    overlap_vis_seed=int(seed),
                )
                elapsed = time.perf_counter() - start_time

                peak_memory_mb = None
                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated(gpu) / 1024 / 1024

                image_path = prompt_dir / f"{suffix}.png"
                save_image(image, image_path)
                images.append(image)

                if log_guidance:
                    trace_payload = {
                        "method": method_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "seed": int(seed),
                        "step_size": float(step_size),
                        "image_path": str(image_path),
                        "time_seconds": elapsed,
                        "peak_memory_mb": peak_memory_mb,
                        "num_attempted_steps": sum(1 for item in trace if item.get("attempted")),
                        "num_updated_steps": sum(1 for item in trace if item.get("updated")),
                        "trace": trace,
                    }
                    with open(prompt_dir / f"{suffix}_guidance_trace.json", "w", encoding="utf-8") as f:
                        json.dump(trace_payload, f, ensure_ascii=False, indent=2)

        if images:
            joined_image = vis_utils.get_image_grid(images)
            joined_name = f"{prompt_id}__{safe_prompt}.png" if prompt_id else f"{safe_prompt}.png"
            joined_image.save(output_root / joined_name)


def load_model(model_path, device=0, force_ddim=True):
    from diffusers import DDIMScheduler
    from diffusion_pipeline import EbamaDiffusionPipeline

    device = (
        torch.device(f"cuda:{device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    pipe = EbamaDiffusionPipeline.from_pretrained(model_path).to(device)
    if force_ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate(pipe, prompt, seed, step_size, attn_res, **guidance_kwargs):
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    result, _ = pipe(
        prompt=prompt,
        generator=generator,
        step_size=step_size,
        attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
        **guidance_kwargs,
    )
    return result["images"][0], list(getattr(pipe, "guidance_trace", []))


def save_image(image, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(file_path)


def _safe_filename(text: str, max_len: int = 180) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {" ", "-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip().replace("  ", " ")
    return name[:max_len] if name else "prompt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a checkered bowl on a red and blue table"
    )
    parser.add_argument("--prompt_id", type=str, default=None)
    parser.add_argument("--method_name", type=str, default="sa_ebama")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_directory", type=str, default="./output")
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="The path to the model.",
    )
    parser.add_argument("--step_size", type=float, default=20.0)
    parser.add_argument("--attn_res", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dist", type=str, default="cos")
    parser.add_argument("--lambda_ours", type=float, default=0.5)

    # SA-EBAMA: Object Separation Regularization
    parser.add_argument("--use_separation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma_sep", type=float, default=0.10)
    parser.add_argument("--sep_temp", type=float, default=0.15)
    parser.add_argument("--sep_tau", type=str, default="percentile", choices=["percentile", "mean_std", "mean"])
    parser.add_argument("--sep_percentile", type=float, default=0.65)

    # SA-EBAMA: Adaptive Timestep Guidance
    parser.add_argument("--use_adaptive_guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--adaptive_mode",
        type=str,
        default="sparse10",
        choices=["fixed25", "ebama", "first_half", "early10", "sparse10", "uniform10", "uniform15", "off"],
    )
    parser.add_argument("--max_guidance_step", type=int, default=25)
    parser.add_argument("--adaptive_loss_threshold", type=float, default=None)
    parser.add_argument("--adaptive_overlap_threshold", type=float, default=0.35)
    parser.add_argument("--adaptive_dynamic_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_ddim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_overlap_vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--overlap_vis_steps",
        type=str,
        default="0,1,2,4,6,8,10,12,16,20",
        help="Comma-separated denoising step indices for averaging object-level attention overlap.",
    )

    args = parser.parse_args()

    main(
        [args.prompt],
        [args.seed],
        args.output_directory,
        args.model_path,
        [args.step_size],
        args.attn_res,
        args.gpu,
        args.dist,
        args.lambda_ours,
        prompt_ids=[args.prompt_id],
        method_name=args.method_name,
        use_separation=args.use_separation,
        gamma_sep=args.gamma_sep,
        sep_temp=args.sep_temp,
        sep_tau=args.sep_tau,
        sep_percentile=args.sep_percentile,
        use_adaptive_guidance=args.use_adaptive_guidance,
        adaptive_mode=args.adaptive_mode,
        max_guidance_step=args.max_guidance_step,
        adaptive_loss_threshold=args.adaptive_loss_threshold,
        adaptive_overlap_threshold=args.adaptive_overlap_threshold,
        adaptive_dynamic_step=args.adaptive_dynamic_step,
        log_guidance=args.log_guidance,
        force_ddim=args.force_ddim,
        save_overlap_vis=args.save_overlap_vis,
        overlap_vis_steps=args.overlap_vis_steps,
    )
