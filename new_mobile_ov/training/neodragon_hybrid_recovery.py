from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from new_mobile_ov.training.neodragon_rollout import (
    downsample_noise_2x,
    prepare_past_conditions,
    pyramid_latents,
    upsample_pyramidal_latent,
)


@dataclass(frozen=True)
class DiTCondition:
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


@dataclass(frozen=True)
class TransitionState:
    start: torch.Tensor
    history: tuple[torch.Tensor, ...]
    unit: int
    stage: int


class StageUnitScaleEMA:
    """EMA of target transition magnitude for stable per-call normalization."""

    def __init__(
        self,
        *,
        num_units: int = 6,
        num_stages: int = 3,
        decay: float = 0.99,
        minimum: float = 0.05,
        maximum: float = 20.0,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("Invalid EMA clamp range.")
        self.num_units = int(num_units)
        self.num_stages = int(num_stages)
        self.decay = float(decay)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.values = torch.ones(self.num_units, self.num_stages, dtype=torch.float64)
        self.initialized = torch.zeros(self.num_units, self.num_stages, dtype=torch.bool)

    def update(self, unit: int, stage: int, magnitude: float) -> float:
        self._validate_key(unit, stage)
        value = min(max(float(magnitude), self.minimum), self.maximum)
        if not self.initialized[unit, stage]:
            self.values[unit, stage] = value
            self.initialized[unit, stage] = True
        else:
            previous = float(self.values[unit, stage])
            self.values[unit, stage] = self.decay * previous + (1.0 - self.decay) * value
        return self.get(unit, stage)

    def get(self, unit: int, stage: int) -> float:
        self._validate_key(unit, stage)
        return min(
            max(float(self.values[unit, stage]), self.minimum),
            self.maximum,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "num_units": self.num_units,
            "num_stages": self.num_stages,
            "decay": self.decay,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "values": self.values.clone(),
            "initialized": self.initialized.clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        values = torch.as_tensor(state["values"], dtype=torch.float64)
        initialized = torch.as_tensor(state["initialized"], dtype=torch.bool)
        expected = (self.num_units, self.num_stages)
        if tuple(values.shape) != expected or tuple(initialized.shape) != expected:
            raise ValueError(
                f"Stage-unit EMA shape mismatch: expected {expected}, got "
                f"{tuple(values.shape)} and {tuple(initialized.shape)}."
            )
        self.values.copy_(values)
        self.initialized.copy_(initialized)

    def _validate_key(self, unit: int, stage: int) -> None:
        if not 0 <= unit < self.num_units:
            raise IndexError(f"unit={unit} is outside [0, {self.num_units}).")
        if not 0 <= stage < self.num_stages:
            raise IndexError(f"stage={stage} is outside [0, {self.num_stages}).")


def balanced_position(
    step: int,
    *,
    num_units: int = 6,
    num_stages: int = 3,
    offset: int = 0,
) -> tuple[int, int]:
    """Cycle through all 18 Hybrid calls identically on every distributed rank."""
    if step < 1:
        raise ValueError("step must be >= 1.")
    index = (step - 1 + int(offset)) % (num_units * num_stages)
    return index // num_stages, index % num_stages


def curriculum_probabilities(
    step: int,
    *,
    parity_steps: int,
    map_end_step: int,
) -> dict[str, float]:
    if step <= parity_steps:
        return {"hybrid_parity": 1.0}
    if step <= map_end_step:
        return {
            "teacher_map": 0.70,
            "hybrid_replay": 0.20,
            "real_endpoint": 0.10,
        }
    return {
        "teacher_map": 0.45,
        "student_replay": 0.30,
        "noisy_history": 0.15,
        "real_endpoint": 0.10,
    }


def sample_curriculum_mode(
    step: int,
    *,
    seed: int,
    parity_steps: int,
    map_end_step: int,
) -> str:
    """Deterministically choose one mode so all DDP ranks execute equal call counts."""
    probabilities = curriculum_probabilities(
        step,
        parity_steps=parity_steps,
        map_end_step=map_end_step,
    )
    rng = random.Random(int(seed) + int(step) * 1_000_003)
    draw = rng.random()
    cumulative = 0.0
    for name, probability in probabilities.items():
        cumulative += probability
        if draw <= cumulative:
            return name
    return next(reversed(probabilities))


def transition_rms(start: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (target.float() - start.float()).square().mean().sqrt()


def normalized_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float | torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    scale_tensor = torch.as_tensor(
        scale,
        device=prediction.device,
        dtype=torch.float32,
    ).clamp_min(1e-6)
    residual = (prediction.float() - target.float()) / scale_tensor
    return torch.sqrt(residual.square() + float(epsilon) ** 2).mean()


def endpoint_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float().flatten(1),
        target.float().flatten(1),
        dim=-1,
    ).mean()


def relative_endpoint_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    start: torch.Tensor,
) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).flatten(1).norm(dim=-1)
    denominator = (target.float() - start.float()).flatten(1).norm(dim=-1)
    return (numerator / denominator.clamp_min(1e-6)).mean()


def hybrid_trust_region_loss(
    student_endpoint: torch.Tensor,
    hybrid_endpoint: torch.Tensor,
    monolithic_endpoint: torch.Tensor,
    *,
    start: torch.Tensor,
    margin_scale: float = 1.0,
    minimum_margin: float = 0.02,
    maximum_margin: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allow only changes justified by the local Monolithic-vs-Hybrid gap."""
    teacher_gap = relative_endpoint_l2(
        monolithic_endpoint,
        hybrid_endpoint,
        start=start,
    ).detach()
    margin = (float(margin_scale) * teacher_gap).clamp(
        min=float(minimum_margin),
        max=float(maximum_margin),
    )
    student_gap = relative_endpoint_l2(
        student_endpoint,
        hybrid_endpoint,
        start=start,
    )
    return F.relu(student_gap - margin).square(), student_gap, margin


def corrupt_history(
    history: tuple[torch.Tensor, ...],
    *,
    strength: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, ...]:
    if strength <= 0.0:
        return history
    outputs = []
    for value in history:
        noise = torch.randn(
            value.shape,
            device=value.device,
            dtype=value.dtype,
            generator=generator,
        )
        scale = value.float().std().clamp_min(1e-4).to(dtype=value.dtype)
        outputs.append(value + float(strength) * scale * noise)
    return tuple(outputs)


def clean_history_for_unit(
    clean_latents: torch.Tensor,
    *,
    unit: int,
    num_stages: int,
) -> tuple[torch.Tensor, ...]:
    if clean_latents.shape[2] < unit + 1:
        raise ValueError(
            f"Need anchor plus {unit} prior units, got T={clean_latents.shape[2]}."
        )
    generated = [
        clean_latents[:, :, index : index + 1]
        for index in range(unit + 1)
    ]
    return tuple(prepare_past_conditions(generated, num_stages)[0])


def clean_endpoint_for_position(
    clean_latents: torch.Tensor,
    *,
    unit: int,
    stage: int,
    num_stages: int,
) -> torch.Tensor:
    target_index = unit + 1
    if clean_latents.shape[2] <= target_index:
        raise ValueError(
            f"Missing clean target unit={unit}; latent T={clean_latents.shape[2]}."
        )
    return pyramid_latents(clean_latents, num_stages)[stage][
        :,
        :,
        target_index : target_index + 1,
    ]


def predict_velocity(
    *,
    dit: torch.nn.Module,
    current: torch.Tensor,
    history: tuple[torch.Tensor, ...],
    condition: DiTCondition,
    timestep: torch.Tensor,
) -> torch.Tensor:
    batch = current.shape[0]
    if condition.uses_guidance:
        if (
            condition.negative_mask is None
            or condition.negative_pooled is None
        ):
            raise ValueError("Guidance requires complete negative condition tensors.")
        model_history = tuple(torch.cat([value, value], dim=0) for value in history)
        model_current = torch.cat([current, current], dim=0)
        tokens = torch.cat([condition.negative_tokens, condition.tokens], dim=0)
        mask = torch.cat([condition.negative_mask, condition.mask], dim=0)
        pooled = torch.cat([condition.negative_pooled, condition.pooled], dim=0)
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


def run_stage_endpoint(
    *,
    dit: torch.nn.Module,
    scheduler,
    current: torch.Tensor,
    history: tuple[torch.Tensor, ...],
    condition: DiTCondition,
    stage: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    timesteps = scheduler.get_stage_timesteps(
        num_steps,
        stage,
        device=current.device,
    )
    sigmas = scheduler.get_stage_sigmas(
        num_steps,
        stage,
        device=current.device,
    )
    first_prediction: torch.Tensor | None = None
    for index, timestep in enumerate(timesteps):
        prediction = predict_velocity(
            dit=dit,
            current=current,
            history=history,
            condition=condition,
            timestep=timestep,
        )
        if first_prediction is None:
            first_prediction = prediction
        current = scheduler.step(
            model_output=prediction,
            sigma=sigmas[index].to(dtype=prediction.dtype),
            sigma_next=sigmas[index + 1].to(dtype=prediction.dtype),
            sample=current,
        ).prev_sample
    if first_prediction is None:
        raise RuntimeError("Stage endpoint produced no prediction.")
    return current, first_prediction


def rollout_state_to_position(
    *,
    actor: torch.nn.Module,
    scheduler,
    anchor: torch.Tensor,
    full_noise: torch.Tensor,
    condition: DiTCondition,
    target_unit: int,
    target_stage: int,
    actor_steps: int,
    num_stages: int = 3,
    generator: torch.Generator | None = None,
) -> TransitionState:
    """Roll one actor to, but not through, a selected Hybrid transition."""
    if anchor.shape[2] != 1:
        raise ValueError(f"Expected one anchor latent, got {tuple(anchor.shape)}.")
    if full_noise.shape[2] <= target_unit + 1:
        raise ValueError(
            f"Noise T={full_noise.shape[2]} does not cover unit={target_unit}."
        )
    low_resolution_noise = downsample_noise_2x(full_noise, num_stages - 1)
    generated = [anchor]

    for unit in range(target_unit + 1):
        stage_histories = prepare_past_conditions(generated, num_stages)
        current = low_resolution_noise[:, :, unit + 1 : unit + 2]
        for stage in range(num_stages):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=generator,
                )
            history = tuple(stage_histories[stage])
            if unit == target_unit and stage == target_stage:
                return TransitionState(
                    start=current,
                    history=history,
                    unit=unit,
                    stage=stage,
                )
            current, _ = run_stage_endpoint(
                dit=actor,
                scheduler=scheduler,
                current=current,
                history=history,
                condition=condition,
                stage=stage,
                num_steps=actor_steps,
            )
        generated.append(current)

    raise RuntimeError("Failed to reach requested Hybrid transition.")


def teacher_forced_state_to_position(
    *,
    actor: torch.nn.Module,
    scheduler,
    clean_latents: torch.Tensor,
    full_noise: torch.Tensor,
    condition: DiTCondition,
    target_unit: int,
    target_stage: int,
    actor_steps: int,
    num_stages: int = 3,
    generator: torch.Generator | None = None,
) -> TransitionState:
    """Build a current-unit state while taking causal history from real latents."""
    if clean_latents.shape[2] <= target_unit + 1:
        raise ValueError(
            f"Clean latent T={clean_latents.shape[2]} does not cover unit={target_unit}."
        )
    if full_noise.shape != clean_latents.shape:
        raise ValueError(
            f"Noise/latent shape mismatch: {tuple(full_noise.shape)} vs "
            f"{tuple(clean_latents.shape)}."
        )

    generated = [
        clean_latents[:, :, index : index + 1]
        for index in range(target_unit + 1)
    ]
    stage_histories = prepare_past_conditions(generated, num_stages)
    low_resolution_noise = downsample_noise_2x(full_noise, num_stages - 1)
    current = low_resolution_noise[:, :, target_unit + 1 : target_unit + 2]
    for stage in range(num_stages):
        if stage > 0:
            current = upsample_pyramidal_latent(
                current,
                orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                gamma=scheduler.config.gamma,
                generator=generator,
            )
        history = tuple(stage_histories[stage])
        if stage == target_stage:
            return TransitionState(
                start=current,
                history=history,
                unit=target_unit,
                stage=stage,
            )
        current, _ = run_stage_endpoint(
            dit=actor,
            scheduler=scheduler,
            current=current,
            history=history,
            condition=condition,
            stage=stage,
            num_steps=actor_steps,
        )

    raise RuntimeError("Failed to reach teacher-forced Hybrid transition.")
