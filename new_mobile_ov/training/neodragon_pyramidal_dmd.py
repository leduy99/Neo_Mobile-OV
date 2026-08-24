"""Stage-wise DMD primitives for reproducing NeoDragon step distillation.

The released NeoDragon checkpoint contains both the multi-step Pyramidal-Flow
teacher and its already step-distilled Hybrid counterpart.  This module only
implements the *training mechanics* described in Sec. 3.4 of the paper: a
multi-step teacher, a one-Euler-step student, and a fake flow model trained on
the student's current endpoint distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass(frozen=True)
class StagePair:
    """Noisy start/end targets for one Pyramidal-Flow stage."""

    start: torch.Tensor
    end: torch.Tensor
    clean: torch.Tensor
    start_sigma: float
    end_sigma: float

    @property
    def flow_target(self) -> torch.Tensor:
        # The scheduler integrates from local sigma=1 to sigma=0.
        return self.start - self.end


@dataclass(frozen=True)
class DMDCondition:
    """Native NeoDragon text condition, with optional classifier-free guidance."""

    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    negative_tokens: torch.Tensor | None = None
    negative_mask: torch.Tensor | None = None
    negative_pooled: torch.Tensor | None = None
    guidance_scale: float = 0.0

    @property
    def uses_guidance(self) -> bool:
        return self.negative_tokens is not None and self.guidance_scale > 0.0


def predict_flow(
    *,
    dit: torch.nn.Module,
    current: torch.Tensor,
    history: tuple[torch.Tensor, ...],
    condition: DMDCondition,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Call the native Pyramidal DiT at one local stage time.

    The teacher uses the paper's CFG values. Student and fake model use the
    same condition path so their DMD difference is not confounded by text
    conditioning.  The released scheduler uses a single time scalar per DDP
    batch, hence ``timestep`` is expanded here.
    """

    batch = current.shape[0]
    if condition.uses_guidance:
        if condition.negative_mask is None or condition.negative_pooled is None:
            raise ValueError("Guidance requires negative tokens, mask, and pooled tensors.")
        model_history = tuple(torch.cat((item, item), dim=0) for item in history)
        model_current = torch.cat((current, current), dim=0)
        tokens = torch.cat((condition.negative_tokens, condition.tokens), dim=0)
        mask = torch.cat((condition.negative_mask, condition.mask), dim=0)
        pooled = torch.cat((condition.negative_pooled, condition.pooled), dim=0)
        model_timestep = timestep.expand(2 * batch)
    else:
        model_history = history
        model_current = current
        tokens = condition.tokens
        mask = condition.mask
        pooled = condition.pooled
        model_timestep = timestep.expand(batch)

    prediction = dit(
        sample=[list(model_history) + [model_current]],
        encoder_hidden_states=tokens,
        encoder_attention_mask=mask,
        pooled_projections=pooled,
        timestep_ratio=model_timestep.to(dtype=model_current.dtype),
    )[0]
    if condition.uses_guidance:
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + float(condition.guidance_scale) * (
            conditional - unconditional
        )
    return prediction


def stage_timestep(
    scheduler,
    *,
    stage: int,
    local_sigma: float | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Map a local stage noise value in ``[0, 1]`` to NeoDragon's DiT time."""

    value = torch.as_tensor(local_sigma, device=device, dtype=torch.float32)
    if value.numel() != 1:
        raise ValueError("This reproduction uses one shared local sigma per DDP batch.")
    if not bool(torch.isfinite(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"local_sigma must be finite and in [0, 1], got {value.item()}")
    # ``timestep_ratios`` indexes the scheduler's *descending* global time
    # array, so mapping it directly would reverse every stage. Ask the release
    # scheduler for its two exact endpoints instead.
    endpoints = scheduler.get_stage_timesteps(2, int(stage), device=device)
    maximum = endpoints[0].to(device=device, dtype=torch.float32)
    minimum = endpoints[-1].to(device=device, dtype=torch.float32)
    return minimum + value * (maximum - minimum)


def _down_then_up(x: torch.Tensor) -> torch.Tensor:
    """Pyramidal ``Up(Down(x, 2), 2)`` at the current latent resolution."""

    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"Pyramid stage must have even H/W, got {tuple(x.shape[-2:])}")
    temporal = x.shape[2]
    image = rearrange(x, "b c t h w -> (b t) c h w")
    image = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)
    image = F.interpolate(image, scale_factor=2.0, mode="nearest")
    return rearrange(image, "(b t) c h w -> b c t h w", t=temporal)


def build_stage_pair(
    *,
    clean: torch.Tensor,
    scheduler,
    stage: int,
    noise: torch.Tensor,
) -> StagePair:
    """Construct Eqs. 17--18 from the NeoDragon report for one stage.

    Stage zero begins at pure noise. Higher-resolution stages begin at a noisy
    upsampled copy of the next coarser clean latent. The same noise realization
    is used at both endpoints, as in the paper's construction.
    """

    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {clean.shape} vs {noise.shape}")
    if int(stage) < 0 or int(stage) >= int(scheduler.config.stages):
        raise ValueError(f"Invalid stage={stage}")
    start_sigma = float(scheduler.orig_start_sigmas[int(stage)])
    end_sigma = float(scheduler.end_sigmas[int(stage)])
    if stage == 0:
        coarse = torch.zeros_like(clean)
    else:
        coarse = _down_then_up(clean)
    start = (1.0 - start_sigma) * coarse + start_sigma * noise
    end = (1.0 - end_sigma) * clean + end_sigma * noise
    return StagePair(
        start=start,
        end=end,
        clean=clean,
        start_sigma=start_sigma,
        end_sigma=end_sigma,
    )


def stage_noisy_student_endpoint(
    *,
    endpoint: torch.Tensor,
    scheduler,
    stage: int,
    local_sigma: float | torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct Eqs. 23--25 around the student's predicted clean endpoint."""

    pair = build_stage_pair(clean=endpoint, scheduler=scheduler, stage=stage, noise=noise)
    tau = torch.as_tensor(local_sigma, device=endpoint.device, dtype=endpoint.dtype)
    probe = (1.0 - tau) * pair.end + tau * pair.start
    return probe, pair.start, pair.end


def cauchy_endpoint_loss(endpoint: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Paper's supervised Cauchy endpoint term, averaged over the batch only."""

    squared_l2 = (endpoint.float() - clean.float()).flatten(1).pow(2).sum(dim=1)
    return torch.log1p(squared_l2).mean()


def rollout_history_probability(
    step: int,
    *,
    warmup_steps: int,
    midpoint_step: int,
    final_step: int,
    midpoint_probability: float,
    final_probability: float,
) -> float:
    """Piecewise-linear probability of using deployed student history."""

    if step < 1:
        raise ValueError("step must be positive")
    if not 0 <= warmup_steps < midpoint_step <= final_step:
        raise ValueError("Expected 0 <= warmup < midpoint <= final step")
    if not 0.0 <= midpoint_probability <= final_probability <= 1.0:
        raise ValueError("History probabilities must satisfy 0 <= midpoint <= final <= 1")
    if step <= warmup_steps:
        return 0.0
    if step <= midpoint_step:
        progress = (step - warmup_steps) / (midpoint_step - warmup_steps)
        return float(progress * midpoint_probability)
    if step <= final_step:
        progress = (step - midpoint_step) / (final_step - midpoint_step)
        return float(
            midpoint_probability
            + progress * (final_probability - midpoint_probability)
        )
    return float(final_probability)


def linear_weight_decay(
    step: int,
    *,
    initial: float,
    final: float,
    decay_steps: int,
) -> float:
    """Linearly decay a non-negative loss weight and then hold it fixed."""

    if step < 1 or decay_steps < 1:
        raise ValueError("step and decay_steps must be positive")
    if initial < 0.0 or final < 0.0:
        raise ValueError("Loss weights must be non-negative")
    progress = min(max((step - 1) / max(decay_steps - 1, 1), 0.0), 1.0)
    return float(initial + progress * (final - initial))


def motion_residual_anchor_loss(
    endpoint: torch.Tensor,
    clean: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """Match the direction and magnitude of the teacher's temporal residual.

    Unlike another endpoint MSE, this loss normalizes the residual direction
    and supervises its RMS magnitude separately. Near-static teacher samples
    contribute only the magnitude term.
    """

    if endpoint.shape != clean.shape or endpoint.shape != previous.shape:
        raise ValueError("endpoint, clean, and previous must have identical shapes")
    predicted = (endpoint.float() - previous.float()).flatten(1)
    target = (clean.float() - previous.float()).flatten(1)
    predicted_rms = predicted.square().mean(dim=1).sqrt()
    target_rms = target.square().mean(dim=1).sqrt()
    magnitude = F.smooth_l1_loss(predicted_rms, target_rms)
    moving = target.norm(dim=1) > 1e-6
    if bool(moving.any()):
        direction = 1.0 - F.cosine_similarity(
            predicted[moving], target[moving], dim=1, eps=1e-6
        ).mean()
    else:
        direction = magnitude.new_zeros(())
    return magnitude + direction


def dmd_sample_weight(
    *,
    teacher_flow: torch.Tensor,
    stage_flow_target: torch.Tensor,
    maximum: float,
    minimum_denominator: float = 1e-6,
) -> torch.Tensor:
    """Inverse L1 teacher-error weighting used by Pyramidal DMD."""

    error = (teacher_flow.float() - stage_flow_target.float()).flatten(1).abs().sum(dim=1)
    return error.clamp_min(float(minimum_denominator)).reciprocal().clamp_max(float(maximum))


def dmd_surrogate_loss(
    *,
    endpoint: torch.Tensor,
    teacher_flow: torch.Tensor,
    fake_flow: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject the DMD gradient while keeping teacher/fake outputs detached.

    The scalar's absolute value has no direct interpretation. Its gradient with
    respect to ``endpoint`` is the paper's teacher-minus-fake direction.
    """

    if sample_weight.ndim != 1 or sample_weight.shape[0] != endpoint.shape[0]:
        raise ValueError("sample_weight must have one value per batch element")
    direction = (teacher_flow.float() - fake_flow.float()) * sample_weight[:, None, None, None, None]
    surrogate = (endpoint.float() * direction.detach()).flatten(1).sum(dim=1).mean()
    direction_rms = direction.flatten(1).pow(2).mean(dim=1).sqrt().mean()
    return surrogate, direction_rms


def student_probe_sigmas() -> tuple[float, float, float, float]:
    """Four evenly spaced local values used for student DMD updates in the paper."""

    return (0.125, 0.375, 0.625, 0.875)
