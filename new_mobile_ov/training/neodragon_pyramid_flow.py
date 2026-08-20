from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PyramidFlowState:
    start: torch.Tensor
    end: torch.Tensor
    noisy: torch.Tensor
    target: torch.Tensor
    stage_noise: torch.Tensor


def _resize_video_latent(
    value: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    batch, channels, frames = value.shape[:3]
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *value.shape[-2:])
    resized = F.interpolate(flat, size=(height, width), mode=mode)
    return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def pyramid_latents(value: torch.Tensor, stages: int = 3) -> list[torch.Tensor]:
    """Build the low-to-high clean latent pyramid used by Pyramidal Flow."""
    if value.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(value.shape)}")
    if stages < 1:
        raise ValueError("stages must be positive")

    outputs = [value]
    current = value
    for _ in range(stages - 1):
        height, width = current.shape[-2] // 2, current.shape[-1] // 2
        current = _resize_video_latent(current, height=height, width=width, mode="bilinear")
        outputs.append(current)
    return list(reversed(outputs))


def correlated_pyramid_noise(noise_high: torch.Tensor, stages: int = 3) -> list[torch.Tensor]:
    """Build official low-to-high noise, preserving variance after each 2x downsample."""
    if noise_high.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(noise_high.shape)}")

    outputs = [noise_high]
    current = noise_high
    for _ in range(stages - 1):
        height, width = current.shape[-2] // 2, current.shape[-1] // 2
        current = 2.0 * _resize_video_latent(
            current,
            height=height,
            width=width,
            mode="bilinear",
        )
        outputs.append(current)
    return list(reversed(outputs))


def _broadcast_batch_scalar(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        value = value.expand(like.shape[0])
    if value.ndim != 1 or value.shape[0] != like.shape[0]:
        raise ValueError(f"Expected one scalar per batch item, got {tuple(value.shape)}")
    return value.reshape(value.shape[0], *([1] * (like.ndim - 1))).to(
        device=like.device,
        dtype=like.dtype,
    )


def build_pyramid_flow_state(
    clean_high: torch.Tensor,
    *,
    stage: int,
    local_sigma: torch.Tensor,
    start_sigma: float,
    end_sigma: float,
    noise_high: torch.Tensor | None = None,
    stages: int = 3,
) -> PyramidFlowState:
    """Construct the exact spatial-stage rectified-flow path used by Pyramidal Flow."""
    if not 0 <= stage < stages:
        raise ValueError(f"stage must be in [0, {stages}), got {stage}")
    clean = pyramid_latents(clean_high, stages=stages)
    if noise_high is None:
        noise_high = torch.randn_like(clean_high)
    noise = correlated_pyramid_noise(noise_high, stages=stages)

    clean_stage = clean[stage]
    stage_noise = noise[stage]
    if stage == 0:
        start = stage_noise
    else:
        previous = clean[stage - 1]
        previous_up = _resize_video_latent(
            previous,
            height=clean_stage.shape[-2],
            width=clean_stage.shape[-1],
            mode="nearest",
        )
        start = float(start_sigma) * stage_noise + (1.0 - float(start_sigma)) * previous_up

    if stage == stages - 1:
        end = clean_stage
    else:
        end = float(end_sigma) * stage_noise + (1.0 - float(end_sigma)) * clean_stage

    sigma = _broadcast_batch_scalar(local_sigma, start)
    noisy = sigma * start + (1.0 - sigma) * end
    return PyramidFlowState(
        start=start,
        end=end,
        noisy=noisy,
        target=start - end,
        stage_noise=stage_noise,
    )


def build_legacy_flow_state(
    clean_high: torch.Tensor,
    *,
    stage: int,
    local_sigma: torch.Tensor,
    noise_high: torch.Tensor | None = None,
    stages: int = 3,
) -> PyramidFlowState:
    """Reproduce the former Mobile-OV objective for controlled comparisons."""
    clean_stage = pyramid_latents(clean_high, stages=stages)[stage]
    if noise_high is None:
        noise_high = torch.randn_like(clean_high)
    stage_noise = pyramid_latents(noise_high, stages=stages)[stage]
    sigma = _broadcast_batch_scalar(local_sigma, clean_stage)
    noisy = sigma * stage_noise + (1.0 - sigma) * clean_stage
    return PyramidFlowState(
        start=stage_noise,
        end=clean_stage,
        noisy=noisy,
        target=stage_noise - clean_stage,
        stage_noise=stage_noise,
    )


def corrupt_history(
    history: list[torch.Tensor],
    *,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Apply the official AR history corruption, using one sigma per sample."""
    outputs = []
    for value in history:
        amount = _broadcast_batch_scalar(sigma, value)
        noise = torch.randn(
            value.shape,
            device=value.device,
            dtype=value.dtype,
            generator=generator,
        )
        outputs.append(amount * noise + (1.0 - amount) * value)
    return outputs


def stage_from_ratio_slot(
    slot: int,
    ratios: tuple[int, ...] = (1, 1, 1),
) -> int:
    """Map a deterministic rank-step slot to a configurable stage ratio."""
    if not ratios or any(int(value) <= 0 for value in ratios):
        raise ValueError("Stage sampling ratios must be positive integers.")
    columns = tuple(stage for stage, count in enumerate(ratios) for _ in range(int(count)))
    return columns[int(slot) % len(columns)]
