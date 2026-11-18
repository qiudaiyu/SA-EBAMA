"""Adaptive timestep guidance utilities for SA-EBAMA.

The functions here are intentionally dependency-light and deterministic. They
support fixed EBAMA-style updates, sparse active-step schedules, loss-triggered
updates, and dynamic step-size rescaling.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set


def get_active_steps(
    mode: str = "sparse10",
    num_inference_steps: int = 50,
    max_guidance_step: int = 25,
    custom_steps: Optional[Sequence[int]] = None,
) -> Set[int]:
    """Return zero-based denoising-loop indices where guidance may be attempted."""
    max_step = max(0, min(int(max_guidance_step), int(num_inference_steps) - 1))

    if custom_steps is not None:
        return {int(s) for s in custom_steps if 0 <= int(s) <= max_step}

    mode = (mode or "fixed25").lower()
    if mode in {"off", "none", "no", "false"}:
        return set()
    if mode in {"fixed25", "ebama", "full", "first_half"}:
        return set(range(max_step + 1))
    if mode == "early10":
        return set(range(min(10, max_step + 1)))
    if mode == "sparse10":
        base = [0, 1, 2, 4, 6, 8, 10, 12, 16, 20]
        return {s for s in base if s <= max_step}
    if mode == "uniform10":
        if max_step == 0:
            return {0}
        count = min(10, max_step + 1)
        return {round(i * max_step / max(count - 1, 1)) for i in range(count)}
    if mode == "uniform15":
        if max_step == 0:
            return {0}
        count = min(15, max_step + 1)
        return {round(i * max_step / max(count - 1, 1)) for i in range(count)}

    raise ValueError(f"Unsupported adaptive guidance mode: {mode}")


def should_attempt_guidance(
    step: int,
    active_steps: Iterable[int],
    max_guidance_step: int = 25,
    last_loss_dict: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, Optional[float]]] = None,
) -> bool:
    """Cheap pre-check used before running an extra UNet forward pass."""
    if step > max_guidance_step:
        return False
    if step in set(active_steps):
        return True

    thresholds = thresholds or {}
    last_loss_dict = last_loss_dict or {}
    loss_thr = thresholds.get("loss")
    sep_thr = thresholds.get("overlap")

    # Re-enter guidance on the next inactive step when the previous measured
    # loss/overlap was high. This gives loss-triggered behavior without paying
    # an attention extraction cost at every timestep.
    if loss_thr is not None and last_loss_dict.get("total", 0.0) > float(loss_thr):
        return True
    if sep_thr is not None and last_loss_dict.get("sep", 0.0) > float(sep_thr):
        return True
    if sep_thr is not None and last_loss_dict.get("overlap", 0.0) > float(sep_thr):
        return True
    return False


def should_update_latent(
    step: int,
    loss_dict: Dict[str, float],
    thresholds: Optional[Dict[str, Optional[float]]] = None,
    active_steps: Optional[Iterable[int]] = None,
    max_guidance_step: int = 25,
    always_update_active: bool = True,
) -> bool:
    """Decide whether to run backward/update after the current loss is known."""
    if step > max_guidance_step:
        return False
    active = set(active_steps or [])
    if always_update_active and step in active:
        return True

    thresholds = thresholds or {}
    loss_thr = thresholds.get("loss")
    sep_thr = thresholds.get("overlap")
    if loss_thr is None and sep_thr is None:
        return step in active if active else True

    if loss_thr is not None and loss_dict.get("total", 0.0) > float(loss_thr):
        return True
    if sep_thr is not None and loss_dict.get("sep", 0.0) > float(sep_thr):
        return True
    if sep_thr is not None and loss_dict.get("overlap", 0.0) > float(sep_thr):
        return True
    return False


def update_moving_average(prev: Optional[float], value: float, momentum: float = 0.9) -> float:
    if prev is None:
        return float(value)
    return float(momentum) * float(prev) + (1.0 - float(momentum)) * float(value)


def dynamic_step_size(
    base_step_size: float,
    step: int,
    loss_value: float,
    loss_ma: Optional[float],
    max_guidance_step: int = 25,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    decay: str = "linear",
) -> float:
    """Loss-aware and timestep-decayed latent update size."""
    base = float(base_step_size)
    if base == 0:
        return 0.0

    if loss_ma is None or abs(loss_ma) < 1e-8:
        loss_scale = 1.0
    else:
        loss_scale = abs(float(loss_value)) / (abs(float(loss_ma)) + 1e-8)
        loss_scale = max(float(min_scale), min(float(max_scale), loss_scale))

    progress = min(max(float(step) / max(float(max_guidance_step), 1.0), 0.0), 1.0)
    if decay == "linear":
        time_scale = 1.0 - 0.5 * progress
    elif decay == "cosine":
        import math

        time_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        time_scale = max(0.25, time_scale)
    else:
        time_scale = 1.0

    return base * loss_scale * time_scale
