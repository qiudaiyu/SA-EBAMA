"""Read per-prompt generated images and their original conditioning text."""
import json
from pathlib import Path


def image_groups(root: Path):
    """Yield the output directory identifier, original prompt and image paths."""
    metadata = root / "run_metadata.json"
    prompt_by_dir = {}
    if metadata.is_file():
        run = json.loads(metadata.read_text(encoding="utf-8"))
        for prompt, prompt_id in zip(run["prompts"], run["prompt_ids"]):
            name = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt)
            name = name.strip().replace("  ", " ")[:180] or "prompt"
            prompt_by_dir[f"{prompt_id}__{name}" if prompt_id else name] = prompt
    groups = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not images:
            continue
        prompt = prompt_by_dir.get(folder.name)
        traces = sorted(folder.glob("*_guidance_trace.json"))
        if prompt is None and traces:
            prompt = json.loads(traces[0].read_text(encoding="utf-8"))["prompt"]
        if prompt is None:
            if "__" in folder.name or metadata.exists():
                raise ValueError(f"Missing original prompt for {folder.name}")
            prompt = folder.name
        groups.append((folder.name, prompt, images))
    if not groups:
        raise ValueError(f"No generated images in prompt directories under {root}")
    return groups
