"""Object Separation Regularization for EBAMA/SA-EBAMA.

This module implements an instance-level soft overlap penalty on cross-attention
maps. It is training-free and differentiable w.r.t. the latent-dependent
attention maps, so it can be added to EBAMA's inference-time latent guidance.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import torch

IndexLike = Union[int, Sequence[int], Sequence[Sequence[int]]]


def _flatten_indices(indices: IndexLike) -> List[int]:
    """Flatten possibly nested token indices into a unique ordered list."""
    if indices is None:
        return []
    if isinstance(indices, int):
        return [indices]

    flattened: List[int] = []
    for item in indices:
        if item is None:
            continue
        if isinstance(item, int):
            flattened.append(item)
        elif isinstance(item, (list, tuple)):
            flattened.extend(_flatten_indices(item))
        else:
            # Ignore unsupported index-like entries instead of crashing inside
            # long prompt sweeps.
            continue

    unique: List[int] = []
    for idx in flattened:
        if idx not in unique:
            unique.append(idx)
    return unique


def _aggregate_attention_for_object(
    attention_maps: Sequence[torch.Tensor], object_index: IndexLike
) -> torch.Tensor | None:
    """Return one spatial attention map for a possibly multi-wordpiece object."""
    flat = [idx for idx in _flatten_indices(object_index) if 0 <= idx < len(attention_maps)]
    if not flat:
        return None
    maps = [attention_maps[idx].float() for idx in flat]
    if len(maps) == 1:
        return maps[0]
    # Mean aggregation is smoother than max aggregation and keeps gradients
    # distributed across wordpieces for compound nouns.
    return torch.stack(maps, dim=0).mean(dim=0)


def normalize_attention(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize one attention map to [0, 1] for stable mask construction."""
    attn = attn.float()
    attn_min = attn.amin()
    attn_max = attn.amax()
    return (attn - attn_min) / (attn_max - attn_min + eps)


def soft_attention_mask(
    attn: torch.Tensor,
    tau: str | float = "percentile",
    temp: float = 0.15,
    percentile: float = 0.65,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert an attention map into a differentiable soft foreground mask.

    Args:
        attn: Spatial attention map with shape H x W.
        tau: ``"percentile"``, ``"mean_std"`` or a numeric threshold in [0, 1].
        temp: Sigmoid temperature. Smaller values make the mask harder.
        percentile: Quantile used when ``tau="percentile"``.
    """
    norm_attn = normalize_attention(attn, eps=eps)
    temp = max(float(temp), eps)

    if isinstance(tau, str):
        tau_key = tau.lower()
        flat = norm_attn.reshape(-1)
        if tau_key == "percentile":
            threshold = torch.quantile(flat.detach(), float(percentile))
        elif tau_key in {"mean_std", "mean+std"}:
            threshold = (flat.detach().mean() + 0.5 * flat.detach().std()).clamp(0.05, 0.95)
        elif tau_key == "mean":
            threshold = flat.detach().mean()
        else:
            raise ValueError(f"Unsupported tau mode: {tau}")
    else:
        threshold = torch.tensor(float(tau), dtype=norm_attn.dtype, device=norm_attn.device)

    return torch.sigmoid((norm_attn - threshold) / temp)


def soft_iou(mask_i: torch.Tensor, mask_j: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Soft IoU between two soft masks. Larger means stronger spatial overlap."""
    intersection = (mask_i * mask_j).sum()
    union = mask_i.sum() + mask_j.sum() - intersection
    return intersection / (union + eps)


def object_separation_loss(
    attention_maps: Sequence[torch.Tensor],
    object_indices: Iterable[IndexLike],
    tau: str | float = "percentile",
    temp: float = 0.15,
    percentile: float = 0.65,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean pairwise SoftIoU over object attention masks.

    Minimizing this term explicitly discourages different object tokens from
    occupying the same spatial attention region, complementing EBAMA's original
    object-attribute alignment loss.
    """
    device = attention_maps[0].device if attention_maps else torch.device("cpu")
    masks: List[torch.Tensor] = []
    for obj_idx in object_indices:
        attn = _aggregate_attention_for_object(attention_maps, obj_idx)
        if attn is None:
            continue
        masks.append(soft_attention_mask(attn, tau=tau, temp=temp, percentile=percentile, eps=eps))

    if len(masks) < 2:
        return torch.zeros((), device=device)

    pair_losses: List[torch.Tensor] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            pair_losses.append(soft_iou(masks[i], masks[j], eps=eps))

    return torch.stack(pair_losses).mean()


@torch.no_grad()
def attention_overlap_ratio(
    attention_maps: Sequence[torch.Tensor],
    object_indices: Iterable[IndexLike],
    tau: str | float = "percentile",
    temp: float = 0.15,
    percentile: float = 0.65,
    eps: float = 1e-8,
) -> float:
    """Detached diagnostic version of object_separation_loss."""
    if not attention_maps:
        return 0.0
    oor = object_separation_loss(
        attention_maps,
        object_indices,
        tau=tau,
        temp=temp,
        percentile=percentile,
        eps=eps,
    )
    return float(oor.detach().cpu())
