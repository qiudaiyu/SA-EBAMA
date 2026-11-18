"""Save attention masks, overlap measurements and preview images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from losses.separation_loss import soft_attention_mask, soft_iou

IndexLike = Union[int, Sequence[int], Sequence[Sequence[int]]]

DEFAULT_VIS_STEPS = [0, 1, 2, 4, 6, 8, 10, 12, 16, 20]


def parse_vis_steps(steps=None) -> List[int]:
    if steps is None:
        return DEFAULT_VIS_STEPS
    if isinstance(steps, str):
        return [int(x.strip()) for x in steps.split(",") if x.strip()]
    return [int(x) for x in steps]


def _flatten_indices(indices: IndexLike) -> List[int]:
    if indices is None:
        return []
    if isinstance(indices, int):
        return [indices]

    flat = []
    for item in indices:
        if item is None:
            continue
        if isinstance(item, int):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            flat.extend(_flatten_indices(item))

    unique = []
    for x in flat:
        if x not in unique:
            unique.append(x)
    return unique


def _aggregate_attention_for_object(attention_maps, object_index):
    flat = [idx for idx in _flatten_indices(object_index) if 0 <= idx < len(attention_maps)]
    if not flat:
        return None

    maps = [attention_maps[idx].float() for idx in flat]
    if len(maps) == 1:
        return maps[0]

    return torch.stack(maps, dim=0).mean(dim=0)


@torch.no_grad()
def build_overlap_snapshot(
    attention_maps,
    object_indices,
    step: int,
    timestep: int,
    tau="percentile",
    temp: float = 0.15,
    percentile: float = 0.70,
    eps: float = 1e-8,
):
    """
    overlap_heatmap = mean_{i<j}(S_i * S_j)
    oor = mean_{i<j} SoftIoU(S_i, S_j)
    """
    object_indices = list(object_indices)

    masks = []
    valid_object_indices = []

    for obj_idx in object_indices:
        attn = _aggregate_attention_for_object(attention_maps, obj_idx)
        if attn is None:
            continue

        mask = soft_attention_mask(
            attn,
            tau=tau,
            temp=temp,
            percentile=percentile,
            eps=eps,
        )

        masks.append(mask.detach().float())
        valid_object_indices.append(_flatten_indices(obj_idx))

    if len(masks) < 2:
        return None

    pair_overlap_maps = []
    pair_oor_values = []
    pair_indices = []

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter_map = masks[i] * masks[j]
            pair_overlap_maps.append(inter_map)
            pair_oor_values.append(soft_iou(masks[i], masks[j], eps=eps))
            pair_indices.append([i, j])

    overlap_heatmap = torch.stack(pair_overlap_maps, dim=0).mean(dim=0)
    oor = torch.stack(pair_oor_values, dim=0).mean()

    return {
        "step": int(step),
        "timestep": int(timestep),
        "object_indices": valid_object_indices,
        "pair_indices": pair_indices,
        "overlap": overlap_heatmap.detach().cpu().numpy().astype(np.float32),
        "masks": torch.stack(masks, dim=0).detach().cpu().numpy().astype(np.float32),
        "pair_oor": np.array(
            [float(x.detach().cpu()) for x in pair_oor_values],
            dtype=np.float32,
        ),
        "oor": float(oor.detach().cpu()),
    }


def save_overlap_summary(
    records,
    output_dir,
    prompt,
    method,
    prompt_id=None,
    seed=None,
    vis_steps=None,
    sep_tau="percentile",
    sep_temp=0.15,
    sep_percentile=0.70,
    image: Optional[Image.Image] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "prompt": prompt,
        "prompt_id": prompt_id,
        "seed": int(seed) if seed is not None else None,
        "method": method,
        "vis_steps_requested": list(vis_steps or DEFAULT_VIS_STEPS),
        "sep_tau": sep_tau,
        "sep_temp": float(sep_temp),
        "sep_percentile": float(sep_percentile),
        "num_records": len(records),
    }

    if image is not None:
        image.save(output_dir / "generated_image_for_vis.png")

    if not records:
        meta["warning"] = "No valid overlap records were collected. Check whether the prompt has at least two detected object tokens."
        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    overlaps = np.stack([r["overlap"] for r in records], axis=0).astype(np.float32)
    avg_overlap = overlaps.mean(axis=0).astype(np.float32)

    oor_values = np.array([r["oor"] for r in records], dtype=np.float32)
    avg_oor = float(oor_values.mean())

    np.save(output_dir / "avg_overlap_heatmap.npy", avg_overlap)
    np.savez_compressed(
        output_dir / "step_overlap_records.npz",
        overlaps=overlaps,
        oor_values=oor_values,
        steps=np.array([r["step"] for r in records], dtype=np.int32),
        timesteps=np.array([r["timestep"] for r in records], dtype=np.int32),
    )

    masks = [r["masks"] for r in records]
    if len({m.shape for m in masks}) == 1:
        avg_masks = np.stack(masks, axis=0).mean(axis=0).astype(np.float32)
        np.savez_compressed(output_dir / "avg_object_masks.npz", masks=avg_masks)

    meta.update(
        {
            "steps_collected": [int(r["step"]) for r in records],
            "timesteps_collected": [int(r["timestep"]) for r in records],
            "avg_oor": avg_oor,
            "min_step_oor": float(oor_values.min()),
            "max_step_oor": float(oor_values.max()),
            "object_indices": records[0].get("object_indices", []),
            "pair_indices": records[0].get("pair_indices", []),
        }
    )

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # The preview uses a per-run color scale.
    try:
        import matplotlib.pyplot as plt

        vmax = float(max(avg_overlap.max(), 1e-8))
        plt.figure(figsize=(3, 3))
        plt.imshow(avg_overlap, cmap="jet", interpolation="bilinear", vmin=0.0, vmax=vmax)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_dir / "overlap_heatmap_preview.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception as e:
        meta["preview_warning"] = str(e)

    return meta
