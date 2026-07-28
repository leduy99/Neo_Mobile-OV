from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from new_mobile_ov.training.neodragon_objectives import flat_cosine_distance


@dataclass
class RolloutDistillationResult:
    mse: torch.Tensor
    cosine: torch.Tensor
    stage_mse: tuple[torch.Tensor, ...]
    unit_mse: tuple[torch.Tensor, ...]
    calls: int
    generated_latents: torch.Tensor


def pyramid_latents(x: torch.Tensor, num_stages: int) -> list[torch.Tensor]:
    """Differentiable pyramid builder for causal history latents."""
    values = [x]
    temporal, height, width = x.shape[-3:]
    for _ in range(num_stages - 1):
        height //= 2
        width //= 2
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.interpolate(x, size=(height, width), mode="bilinear")
        x = rearrange(x, "(b t) c h w -> b c t h w", t=temporal)
        values.append(x)
    return list(reversed(values))


def prepare_past_conditions(
    generated_latents: list[torch.Tensor],
    num_stages: int,
) -> list[list[torch.Tensor]]:
    """Build NeoDragon history without the inference helper's no-grad boundary."""
    if not generated_latents:
        return [[] for _ in range(num_stages)]

    frames_per_unit = generated_latents[0].shape[2]
    unit_index = len(generated_latents)
    history = pyramid_latents(torch.cat(generated_latents, dim=2), num_stages)
    outputs: list[list[torch.Tensor]] = []
    for stage in range(num_stages):
        stage_input = [history[stage][:, :, -frames_per_unit:]]
        current_stage = stage
        units_covered = 1
        while units_covered < unit_index:
            current_stage = max(current_stage - 1, 0)
            if current_stage == 0:
                break
            units_covered += 1
            begin = -(units_covered * frames_per_unit)
            end = -((units_covered - 1) * frames_per_unit)
            stage_input.append(history[current_stage][:, :, begin:end])
        if current_stage == 0 and units_covered < unit_index:
            stage_input.append(
                history[0][:, :, : -(units_covered * frames_per_unit)]
            )
        outputs.append(list(reversed(stage_input)))
    return outputs


def downsample_noise_2x(latents: torch.Tensor, times: int) -> torch.Tensor:
    """Match NeoDragon's initial low-resolution noise preparation."""
    temporal, height, width = latents.shape[-3:]
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    for _ in range(times):
        height //= 2
        width //= 2
        latents = F.interpolate(
            latents,
            size=(height, width),
            mode="bilinear",
        ) * 2.0
    return rearrange(latents, "(b t) c h w -> b c t h w", t=temporal)


def _block_noise(
    reference: torch.Tensor,
    gamma: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Vectorized form of NeoDragon's correlated 2x2 corrective noise."""
    batch, channels, temporal, height, width = reference.shape
    if height % 2 or width % 2:
        raise ValueError(f"Corrective block noise requires even H/W, got {height}x{width}")
    covariance = (
        torch.eye(4, device=reference.device, dtype=torch.float32) * (1.0 + gamma)
        - torch.ones(4, 4, device=reference.device, dtype=torch.float32) * gamma
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    transform = eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt())
    blocks = batch * channels * temporal * (height // 2) * (width // 2)
    iid = torch.randn(
        blocks,
        4,
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )
    noise = iid @ transform.T
    noise = rearrange(
        noise,
        "(b c t h w) (p q) -> b c t (h p) (w q)",
        b=batch,
        c=channels,
        t=temporal,
        h=height // 2,
        w=width // 2,
        p=2,
        q=2,
    )
    return noise.to(dtype=reference.dtype)


def upsample_pyramidal_latent(
    latents: torch.Tensor,
    *,
    orig_sigma: float,
    gamma: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Differentiable coarse-to-fine transition with corrective block noise."""
    temporal = latents.shape[2]
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    latents = F.interpolate(latents, scale_factor=2, mode="nearest")
    latents = rearrange(latents, "(b t) c h w -> b c t h w", t=temporal)

    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_sigma) + orig_sigma)
    beta = alpha * (1 - orig_sigma) / math.sqrt(gamma)
    return alpha * latents + beta * _block_noise(
        latents,
        gamma,
        generator=generator,
    )


def rollout_distillation_loss(
    *,
    dit: torch.nn.Module,
    scheduler,
    anchor_latent: torch.Tensor,
    student_tokens: torch.Tensor,
    student_mask: torch.Tensor,
    student_pooled: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_mask: torch.Tensor,
    teacher_pooled: torch.Tensor,
    num_generated_units: int = 6,
    num_stages: int = 3,
    generator: torch.Generator | None = None,
) -> RolloutDistillationResult:
    """Distill all calls of NeoDragon's hybrid autoregressive rollout.

    The teacher is queried at the detached student state. The student state is
    never detached, so losses from later calls backpropagate through every
    earlier scheduler update and causal-history dependency.
    """
    if anchor_latent.shape[2] != 1:
        raise ValueError(
            f"Expected one anchor latent frame, got {tuple(anchor_latent.shape)}"
        )
    if num_generated_units < 1 or num_stages < 1:
        raise ValueError("num_generated_units and num_stages must be positive")

    batch, channels, _, height, width = anchor_latent.shape
    full_noise = torch.randn(
        batch,
        channels,
        num_generated_units + 1,
        height,
        width,
        device=anchor_latent.device,
        dtype=anchor_latent.dtype,
        generator=generator,
    )
    low_resolution_noise = downsample_noise_2x(full_noise, num_stages - 1)
    generated: list[torch.Tensor] = [anchor_latent.detach()]
    mse_by_call: list[torch.Tensor] = []
    cosine_by_call: list[torch.Tensor] = []
    mse_by_stage: list[list[torch.Tensor]] = [[] for _ in range(num_stages)]
    mse_by_unit: list[list[torch.Tensor]] = [[] for _ in range(num_generated_units)]

    dit.eval()
    for unit in range(num_generated_units):
        past_conditions = prepare_past_conditions(generated, num_stages)
        current = low_resolution_noise[:, :, unit + 1 : unit + 2]
        for stage in range(num_stages):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=generator,
                )

            timesteps = scheduler.get_stage_timesteps(1, stage, device=current.device)
            sigmas = scheduler.get_stage_sigmas(1, stage, device=current.device)
            timestep = timesteps[0].expand(batch).to(dtype=current.dtype)
            sigma = sigmas[0].to(dtype=current.dtype)
            sigma_next = sigmas[1].to(dtype=current.dtype)
            stage_input = past_conditions[stage] + [current]

            with torch.no_grad():
                teacher_prediction = dit(
                    sample=[[value.detach() for value in stage_input]],
                    encoder_hidden_states=teacher_tokens,
                    encoder_attention_mask=teacher_mask,
                    pooled_projections=teacher_pooled,
                    timestep_ratio=timestep,
                )[0]

            student_prediction = dit(
                sample=[stage_input],
                encoder_hidden_states=student_tokens,
                encoder_attention_mask=student_mask,
                pooled_projections=student_pooled,
                timestep_ratio=timestep,
            )[0]
            call_mse = F.mse_loss(
                student_prediction.float(),
                teacher_prediction.float(),
            )
            call_cosine = flat_cosine_distance(
                student_prediction,
                teacher_prediction,
            )
            mse_by_call.append(call_mse)
            cosine_by_call.append(call_cosine)
            mse_by_stage[stage].append(call_mse)
            mse_by_unit[unit].append(call_mse)
            current = scheduler.step(
                model_output=student_prediction,
                sigma=sigma,
                sigma_next=sigma_next,
                sample=current,
            ).prev_sample
        generated.append(current)

    return RolloutDistillationResult(
        mse=torch.stack(mse_by_call).mean(),
        cosine=torch.stack(cosine_by_call).mean(),
        stage_mse=tuple(torch.stack(values).mean() for values in mse_by_stage),
        unit_mse=tuple(torch.stack(values).mean() for values in mse_by_unit),
        calls=len(mse_by_call),
        generated_latents=torch.cat(generated, dim=2),
    )
