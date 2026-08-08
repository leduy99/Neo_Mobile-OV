from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from new_mobile_ov.training.neodragon_hybrid_recovery import (
    DiTCondition,
    run_stage_endpoint,
)
from new_mobile_ov.training.neodragon_rollout import (
    downsample_noise_2x,
    prepare_past_conditions,
    upsample_pyramidal_latent,
)


@dataclass(frozen=True)
class OPSDTransitionState:
    """One exact Hybrid call with generated and privileged causal histories."""

    start: torch.Tensor
    student_history: tuple[torch.Tensor, ...]
    teacher_history: tuple[torch.Tensor, ...]
    unit: int
    stage: int


def balanced_joint_position(
    step: int,
    *,
    num_units: int,
    num_stages: int,
    offset: int = 0,
) -> tuple[int, int]:
    """Cycle over every deployed call, including early preservation calls."""
    if step < 1:
        raise ValueError("step must be >= 1.")
    if num_units < 1 or num_stages < 1:
        raise ValueError("num_units and num_stages must be positive.")
    index = (step - 1 + int(offset)) % (num_units * num_stages)
    return index // num_stages, index % num_stages


def balanced_opsd_position(
    step: int,
    *,
    min_unit: int,
    num_units: int,
    num_stages: int,
    offset: int = 0,
) -> tuple[int, int]:
    """Cycle over positions that have enough old history for OPSD correction."""
    if step < 1:
        raise ValueError("step must be >= 1.")
    if not 0 <= min_unit < num_units:
        raise ValueError(
            f"min_unit={min_unit} must be within [0, {num_units})."
        )
    if num_stages < 1:
        raise ValueError("num_stages must be positive.")
    positions = (num_units - min_unit) * num_stages
    index = (step - 1 + int(offset)) % positions
    return min_unit + index // num_stages, index % num_stages


def build_privileged_units(
    clean_latents: torch.Tensor,
    generated_units: list[torch.Tensor],
    *,
    keep_recent_generated: int,
) -> list[torch.Tensor]:
    """Replace old generated units by GT while retaining recent AR context.

    The current denoising state is never replaced. Only the causal history used
    by the frozen teacher changes, matching the privileged-context principle in
    OPSD-V without treating real video as a reconstruction target.
    """
    if keep_recent_generated < 1:
        raise ValueError("keep_recent_generated must be >= 1.")
    if clean_latents.shape[2] < len(generated_units):
        raise ValueError(
            f"Clean latent T={clean_latents.shape[2]} cannot cover "
            f"{len(generated_units)} history units."
        )
    replace_before = max(len(generated_units) - keep_recent_generated, 0)
    privileged: list[torch.Tensor] = []
    for index, generated in enumerate(generated_units):
        if index < replace_before:
            privileged.append(clean_latents[:, :, index : index + 1])
        else:
            privileged.append(generated)
    return privileged


def rollout_opsd_state_to_position(
    *,
    actor: torch.nn.Module,
    scheduler,
    clean_latents: torch.Tensor,
    full_noise: torch.Tensor,
    condition: DiTCondition,
    target_unit: int,
    target_stage: int,
    keep_recent_generated: int = 1,
    num_stages: int = 3,
    generator: torch.Generator | None = None,
) -> OPSDTransitionState:
    """Run the Student policy to one deployed 1-1-1 call under no grad.

    The returned Student and teacher histories differ only in older temporal
    context. Both branches share the exact same stage-start latent.
    """
    if clean_latents.shape[2] <= target_unit + 1:
        raise ValueError(
            f"Clean latent T={clean_latents.shape[2]} does not cover "
            f"target unit={target_unit}."
        )
    if full_noise.shape != clean_latents.shape:
        raise ValueError(
            f"Noise/latent shape mismatch: {tuple(full_noise.shape)} vs "
            f"{tuple(clean_latents.shape)}."
        )
    if not 0 <= target_stage < num_stages:
        raise ValueError(
            f"target_stage={target_stage} must be within [0, {num_stages})."
        )

    low_resolution_noise = downsample_noise_2x(full_noise, num_stages - 1)
    generated = [clean_latents[:, :, :1]]

    for unit in range(target_unit + 1):
        student_histories = prepare_past_conditions(generated, num_stages)
        privileged_units = build_privileged_units(
            clean_latents,
            generated,
            keep_recent_generated=keep_recent_generated,
        )
        teacher_histories = prepare_past_conditions(privileged_units, num_stages)
        current = low_resolution_noise[:, :, unit + 1 : unit + 2]

        for stage in range(num_stages):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=generator,
                )
            if unit == target_unit and stage == target_stage:
                return OPSDTransitionState(
                    start=current,
                    student_history=tuple(student_histories[stage]),
                    teacher_history=tuple(teacher_histories[stage]),
                    unit=unit,
                    stage=stage,
                )
            current, _ = run_stage_endpoint(
                dit=actor,
                scheduler=scheduler,
                current=current,
                history=tuple(student_histories[stage]),
                condition=condition,
                stage=stage,
                num_steps=1,
            )
        generated.append(current.detach())

    raise RuntimeError("Failed to reach the requested OPSD transition.")


def history_relative_l2(
    student_history: tuple[torch.Tensor, ...],
    teacher_history: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student_history) != len(teacher_history) or not student_history:
        raise ValueError("Student and teacher histories must be non-empty and aligned.")
    squared_error = student_history[0].new_zeros((), dtype=torch.float32)
    squared_reference = student_history[0].new_zeros((), dtype=torch.float32)
    count = 0
    for student, teacher in zip(student_history, teacher_history):
        if student.shape != teacher.shape:
            raise ValueError(
                f"History shape mismatch: {tuple(student.shape)} vs "
                f"{tuple(teacher.shape)}."
            )
        squared_error = squared_error + (student.float() - teacher.float()).square().sum()
        squared_reference = squared_reference + teacher.float().square().sum()
        count += teacher.numel()
    rms_error = (squared_error / max(count, 1)).sqrt()
    rms_reference = (squared_reference / max(count, 1)).sqrt()
    return rms_error / rms_reference.clamp_min(1e-6)


def velocity_rms(velocity: torch.Tensor) -> torch.Tensor:
    return velocity.float().square().mean().sqrt()


def normalized_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    scale_tensor = torch.as_tensor(
        scale,
        device=prediction.device,
        dtype=torch.float32,
    ).clamp_min(1e-6)
    return ((prediction.float() - target.float()) / scale_tensor).square().mean()


def velocity_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float().flatten(1),
        target.float().flatten(1),
        dim=-1,
    ).mean()


def relative_velocity_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    scale_tensor = torch.as_tensor(
        scale,
        device=prediction.device,
        dtype=torch.float32,
    ).clamp_min(1e-6)
    return (
        (prediction.float() - target.float()).square().mean(dim=(1, 2, 3, 4)).sqrt()
        / scale_tensor
    ).mean()


def adaptive_base_trust_loss(
    student_velocity: torch.Tensor,
    base_velocity: torch.Tensor,
    privileged_velocity: torch.Tensor,
    *,
    scale: float | torch.Tensor,
    margin_scale: float | torch.Tensor = 1.0,
    minimum_margin: float = 0.01,
    maximum_margin: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permit movement from the base only where clean context justifies it."""
    teacher_gap = relative_velocity_l2(
        privileged_velocity,
        base_velocity,
        scale=scale,
    ).detach()
    margin_scale_tensor = torch.as_tensor(
        margin_scale,
        device=teacher_gap.device,
        dtype=teacher_gap.dtype,
    )
    margin = (margin_scale_tensor * teacher_gap).clamp(
        min=float(minimum_margin),
        max=float(maximum_margin),
    )
    student_gap = relative_velocity_l2(
        student_velocity,
        base_velocity,
        scale=scale,
    )
    loss = F.relu(student_gap - margin).square()
    return loss, student_gap, teacher_gap, margin


def teacher_advantage_gate(
    base_error: torch.Tensor,
    privileged_error: torch.Tensor,
    *,
    margin: float = 0.0,
    ramp: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use privileged supervision only when it improves the clean diagnostic.

    The clean endpoint selects whether a teacher correction is trustworthy; it
    is never used as a regression target. This is important for a pruned,
    one-step transition map where cleaner history need not help every call.
    """
    if ramp <= 0.0:
        raise ValueError("Teacher advantage ramp must be positive.")
    relative_gain = (base_error.detach() - privileged_error.detach()) / base_error.detach().clamp_min(1e-6)
    gate = ((relative_gain - float(margin)) / float(ramp)).clamp(0.0, 1.0)
    return gate, relative_gain
