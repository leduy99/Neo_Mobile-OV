#!/usr/bin/env python
"""Localize motion loss in the NeoDragon Pyramidal-DMD student.

This audit uses the exact 320p NeoDragon components and compares four
controlled rollouts: the multi-step teacher, DMD under teacher-forced history,
DMD under its deployed self-history, and the released Hybrid.  It records
actual teacher stage endpoints, which lets us test whether the trainer's
bilinear pyramid targets match the states consumed by the deployed sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from einops import rearrange
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramidal_dmd import DMDCondition, _down_then_up, predict_flow
from new_mobile_ov.training.neodragon_rollout import (
    downsample_noise_2x,
    prepare_past_conditions,
    pyramid_latents,
    upsample_pyramidal_latent,
)


DEFAULT_PROMPTS = (
    "An astronaut walks slowly across the red surface of Mars as dust blows behind them.",
    "A vintage red car drives along a coastal road at golden hour, the camera tracking beside it.",
    "A wide cinematic view of a waterfall flowing into a misty tropical valley, leaves moving in the wind.",
)


@dataclass
class Trace:
    start: torch.Tensor
    end: torch.Tensor
    history: tuple[torch.Tensor, ...]
    update_rms: float
    flow_rms: float


@dataclass
class Rollout:
    latents: torch.Tensor
    trace: dict[tuple[int, int], Trace]


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Tensor shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    ref = reference.float().reshape(1, -1)
    value = candidate.float().reshape(1, -1)
    difference = value - ref
    return {
        "cosine": float(F.cosine_similarity(ref, value, dim=1).mean().cpu()),
        "relative_l2": float((difference.norm() / ref.norm().clamp_min(1e-12)).cpu()),
        "mse": float(F.mse_loss(value, ref).cpu()),
        "reference_rms": float(ref.square().mean().sqrt().cpu()),
        "candidate_rms": float(value.square().mean().sqrt().cpu()),
    }


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def latent_motion_metrics(latents: torch.Tensor) -> dict[str, float]:
    values = latents.float()
    all_delta = values[:, :, 1:] - values[:, :, :-1]
    video_delta = values[:, :, 2:] - values[:, :, 1:-1]
    return {
        "latent_rms": float(values.square().mean().sqrt().cpu()),
        "all_unit_delta_rms": float(all_delta.square().mean().sqrt().cpu()),
        "video_unit_delta_rms": float(video_delta.square().mean().sqrt().cpu()),
        "motion_energy_ratio": float(
            (video_delta.square().mean() / values[:, :, 1:].square().mean().clamp_min(1e-12)).cpu()
        ),
        "anchor_to_last_rms": float((values[:, :, -1] - values[:, :, 0]).square().mean().sqrt().cpu()),
    }


def decoded_motion_metrics(frames: np.ndarray) -> dict[str, float]:
    if len(frames) < 2:
        raise ValueError("Expected at least two decoded frames.")
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    mae = []
    flow = []
    for previous, current in zip(frames[:-1], frames[1:]):
        mae.append(float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 255.0))
    for previous, current in zip(gray[:-1], gray[1:]):
        dense = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow.append(float(np.linalg.norm(dense, axis=-1).mean()))
    sharpness = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in gray]
    return {
        "frame_mae_mean": float(np.mean(mae)),
        "frame_mae_p95": float(np.percentile(mae, 95)),
        "optical_flow_mean": float(np.mean(flow)),
        "optical_flow_p95": float(np.percentile(flow, 95)),
        "laplacian_variance_mean": float(np.mean(sharpness)),
    }


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> np.ndarray:
    from neodragon.utils.generation_utils import (
        VAE_SCALE_FACTOR,
        VAE_SHIFT_FACTOR,
        VAE_VIDEO_SCALE_FACTOR,
        VAE_VIDEO_SHIFT_FACTOR,
    )

    values = latents.clone()
    values[:, :, :1] = (values[:, :, :1] / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
    if values.shape[2] > 1:
        values[:, :, 1:] = (values[:, :, 1:] / VAE_VIDEO_SCALE_FACTOR) + VAE_VIDEO_SHIFT_FACTOR
    decoded = vae.decode(values).sample
    decoded = rearrange(decoded, "b c t h w -> (b t) h w c")
    return (
        decoded.float().clamp(-1.0, 1.0).add(1.0).mul(127.5).round().byte().cpu().numpy()
    )


def make_condition(
    *,
    text: torch.nn.Module,
    adapter: torch.nn.Module,
    prompt: str,
    negative_prompt: str,
    guidance: float,
    device: torch.device,
) -> DMDCondition:
    tokens, mask, pooled = text([prompt], device)
    tokens = adapter(tokens)
    if guidance <= 0.0:
        return DMDCondition(tokens=tokens, mask=mask, pooled=pooled)
    negative_tokens, negative_mask, negative_pooled = text([negative_prompt], device)
    return DMDCondition(
        tokens=tokens,
        mask=mask,
        pooled=pooled,
        negative_tokens=adapter(negative_tokens),
        negative_mask=negative_mask,
        negative_pooled=negative_pooled,
        guidance_scale=guidance,
    )


def transition_generator(device: torch.device, seed: int, unit: int, stage: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed + 1009 * unit + 97 * stage))


def run_rollout(
    *,
    dit: torch.nn.Module,
    scheduler,
    full_noise: torch.Tensor,
    condition_for_unit,
    first_steps: int,
    video_steps: int,
    transition_seed: int,
    initial_anchor: torch.Tensor | None,
    teacher_history: torch.Tensor | None = None,
) -> Rollout:
    """Run the 3-stage autoregressive schedule and retain deployed states."""

    generated = [] if initial_anchor is None else [initial_anchor]
    low_noise = downsample_noise_2x(full_noise, 2)
    trace: dict[tuple[int, int], Trace] = {}
    units = full_noise.shape[2]
    for unit in range(len(generated), units):
        history_units = (
            [teacher_history[:, :, index : index + 1] for index in range(unit)]
            if teacher_history is not None
            else generated
        )
        histories = prepare_past_conditions(history_units, num_stages=3)
        current = low_noise[:, :, unit : unit + 1]
        steps = first_steps if unit == 0 else video_steps
        condition = condition_for_unit(unit)
        for stage in range(3):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1.0 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=transition_generator(torch.device(current.device), transition_seed, unit, stage),
                )
            start = current.detach().clone()
            timesteps = scheduler.get_stage_timesteps(steps, stage, device=current.device)
            sigmas = scheduler.get_stage_sigmas(steps, stage, device=current.device)
            first_flow = None
            for step in range(steps):
                flow = predict_flow(
                    dit=dit,
                    current=current,
                    history=tuple(histories[stage]),
                    condition=condition,
                    timestep=timesteps[step],
                )
                if first_flow is None:
                    first_flow = flow.detach()
                current = scheduler.step(
                    model_output=flow,
                    sigma=sigmas[step].to(dtype=current.dtype),
                    sigma_next=sigmas[step + 1].to(dtype=current.dtype),
                    sample=current,
                ).prev_sample
            trace[(unit, stage)] = Trace(
                start=start,
                end=current.detach().clone(),
                history=tuple(value.detach().clone() for value in histories[stage]),
                update_rms=float((current.float() - start.float()).square().mean().sqrt().cpu()),
                flow_rms=float(first_flow.float().square().mean().sqrt().cpu()),
            )
        generated.append(current)
    return Rollout(latents=torch.cat(generated, dim=2), trace=trace)


def oracle_local_dmd_metrics(
    *,
    teacher: Rollout,
    student: torch.nn.Module,
    condition: DMDCondition,
    scheduler,
) -> dict[str, object]:
    """Evaluate every DMD one-step endpoint on the true teacher state/history."""

    by_call = []
    by_stage: list[list[dict[str, float]]] = [[], [], []]
    for unit in range(1, teacher.latents.shape[2]):
        for stage in range(3):
            source = teacher.trace[(unit, stage)]
            timestep = scheduler.get_stage_timesteps(1, stage, device=source.start.device)[0]
            student_flow = predict_flow(
                dit=student,
                current=source.start,
                history=source.history,
                condition=condition,
                timestep=timestep,
            )
            student_end = scheduler.step(
                model_output=student_flow,
                sigma=torch.tensor(1.0, device=source.start.device, dtype=source.start.dtype),
                sigma_next=torch.tensor(0.0, device=source.start.device, dtype=source.start.dtype),
                sample=source.start,
            ).prev_sample
            endpoint = tensor_metrics(source.end, student_end)
            by_stage[stage].append(endpoint)
            by_call.append({"unit": unit, "stage": stage, **endpoint})
    return {
        "mean": average([{key: value for key, value in row.items() if key not in {"unit", "stage"}} for row in by_call]),
        "by_stage": {str(stage): average(values) for stage, values in enumerate(by_stage)},
        "by_call": by_call,
    }


def teacher_target_mismatch(teacher: Rollout) -> dict[str, object]:
    """Compare the trainer's reconstructed pairs with real teacher states.

    The synthetic dataset retains only the final full-resolution latent.  The
    current trainer reconstructs every stage target by bilinearly resizing it,
    whereas deployed Pyramidal-Flow enters later stages through the previous
    stage endpoint plus corrective block noise.  Measuring both ends makes
    that otherwise hidden training/deployment mismatch explicit.
    """

    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler

    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    pseudo_pyramid = pyramid_latents(teacher.latents, num_stages=3)
    endpoint_by_call = []
    transition_by_call = []
    endpoint_by_stage: list[list[dict[str, float]]] = [[], [], []]
    transition_by_stage: list[list[dict[str, float]]] = [[], [], []]

    def upsample_2x(value: torch.Tensor) -> torch.Tensor:
        image = rearrange(value, "b c t h w -> (b t) c h w")
        image = F.interpolate(image, scale_factor=2.0, mode="nearest")
        return rearrange(image, "(b t) c h w -> b c t h w", b=value.shape[0], t=value.shape[2])

    for unit in range(1, teacher.latents.shape[2]):
        for stage in range(3):
            trace = teacher.trace[(unit, stage)]
            pseudo_clean = pseudo_pyramid[stage][:, :, unit : unit + 1]
            endpoint_values = tensor_metrics(trace.end, pseudo_clean)
            endpoint_by_stage[stage].append(endpoint_values)
            endpoint_by_call.append({"unit": unit, "stage": stage, **endpoint_values})
            if stage > 0:
                # ``build_stage_pair`` represents the training start as
                # (1 - orig_start_sigma) * Down(Up(final_latent)) + noise.
                # The deployed sampler instead applies the Pyramidal-Flow
                # alpha correction to the previous stage endpoint before
                # adding correlated block noise. Compare just those two
                # deterministic bases; the noise draw is intentionally not a
                # confound here.
                start_sigma = float(scheduler.orig_start_sigmas[stage])
                trainer_base = (1.0 - start_sigma) * _down_then_up(pseudo_clean)
                orig_sigma = 1.0 - start_sigma
                alpha = 1.0 / (
                    math.sqrt(1.0 + (1.0 / scheduler.gamma)) * (1.0 - orig_sigma)
                    + orig_sigma
                )
                deployed_base = alpha * upsample_2x(teacher.trace[(unit, stage - 1)].end)
                transition_values = tensor_metrics(deployed_base, trainer_base)
                transition_by_stage[stage].append(transition_values)
                transition_by_call.append({"unit": unit, "stage": stage, **transition_values})
    return {
        "meaning": "actual multi-step stage states versus final-latent bilinear reconstruction used by the current trainer",
        "endpoint": {
            "mean": average(
                [{key: value for key, value in row.items() if key not in {"unit", "stage"}} for row in endpoint_by_call]
            ),
            "by_stage": {str(stage): average(values) for stage, values in enumerate(endpoint_by_stage)},
            "by_call": endpoint_by_call,
        },
        "stage_transition_base": {
            "meaning": "deployed alpha-corrected upsampled previous-stage endpoint versus the current trainer's (1-orig_start_sigma)*Down(Up(final-latent)) base; corrective block noise is intentionally excluded",
            "mean": average(
                [{key: value for key, value in row.items() if key not in {"unit", "stage"}} for row in transition_by_call]
            ),
            "by_stage": {str(stage): average(values) for stage, values in enumerate(transition_by_stage) if values},
            "by_call": transition_by_call,
        },
    }


def checkpoint_update_metrics(
    reference: torch.nn.Module,
    candidate: torch.nn.Module | dict[str, torch.Tensor],
) -> dict[str, object]:
    """Measure DMD parameter drift relative to the released multistep init."""

    teacher_state = reference.state_dict()
    student_state = candidate.state_dict() if isinstance(candidate, torch.nn.Module) else candidate
    total_delta = 0.0
    total_reference = 0.0
    groups: dict[str, list[float]] = {}
    for key, reference in teacher_state.items():
        candidate_value = student_state[key]
        if not torch.is_floating_point(reference):
            continue
        # The checkpoint stays on CPU while the initial model is on CUDA.
        # Move one tensor at a time so the diagnostic does not duplicate a
        # full 3 GB DiT state on the GPU.
        candidate_value = candidate_value.to(device=reference.device, dtype=reference.dtype)
        delta = float((candidate_value.float() - reference.float()).square().sum().cpu())
        base = float(reference.float().square().sum().cpu())
        total_delta += delta
        total_reference += base
        group = key.split(".", 1)[0]
        bucket = groups.setdefault(group, [0.0, 0.0])
        bucket[0] += delta
        bucket[1] += base
    return {
        "global_relative_l2": math.sqrt(total_delta / max(total_reference, 1e-12)),
        "top_level_relative_l2": {
            name: math.sqrt(delta / max(base, 1e-12)) for name, (delta, base) in groups.items()
        },
    }


def encode_ssd_anchor(
    *,
    pipeline,
    vae: torch.nn.Module,
    prompt: str,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Image.Image]:
    from neodragon.utils.generation_utils import VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR, _pil_to_numpy

    image = pipeline(
        prompt=prompt,
        num_images_per_prompt=1,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0].convert("RGB")
    resized = image.resize((width, height))
    values = torch.from_numpy(_pil_to_numpy(resized)).to(device=device, dtype=dtype)
    values = rearrange(values, "(b t) h w c -> b c t h w", b=1, t=1)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    encoded = vae.encode(values).latent_dist.sample()
    return (encoded - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR, image


def load_models(
    cfg,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    include_hybrid: bool,
    include_ssd_anchor: bool,
) -> dict[str, object]:
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schedule") not in {
        "hybrid_1-1-1_video_units_only",
        "pyramidal_1-1-1_all_native_units",
    }:
        raise ValueError(f"Not a Pyramidal-DMD checkpoint: {checkpoint}")
    checkpoint_step = int(payload.get("step", -1))
    adapter_id = str(payload.get("context_adapter_id", MULTISTEP_CONTEXT_ADAPTER_ID))
    dit_id = str(payload.get("teacher_dit_id", MULTISTEP_DIT_ID))
    student_state = payload.pop("student")
    del payload

    print("Loading frozen NeoDragon text stack...", flush=True)
    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    multi_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{adapter_id}",
        torch_dtype=dtype,
    ).to(device).eval()
    print("Loading frozen multistep teacher DiT...", flush=True)
    teacher = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}", torch_dtype=dtype
    ).to(device).eval()
    print("Loading DMD student DiT...", flush=True)
    student = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}", torch_dtype=dtype
    ).to(device).eval()
    student.load_state_dict(student_state, strict=True)
    del student_state
    hybrid_adapter = None
    hybrid = None
    if include_hybrid:
        print("Loading released Hybrid DiT control...", flush=True)
        hybrid_adapter = ContextAdapter.from_pretrained(
            f"{local_model_path}/context_adapter", torch_dtype=dtype
        ).to(device).eval()
        hybrid = PyramidMMDiT.from_pretrained(
            f"{local_model_path}/diffusion_transformer_320p", torch_dtype=dtype
        ).to(device).eval()
    print("Loading VAE...", flush=True)
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/causal_video_vae", torch_dtype=dtype
    ).to(device).eval()
    first_frame = None
    if include_ssd_anchor:
        print("Loading SSD1B anchor pipeline...", flush=True)
        first_frame = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(
            local_model_path, torch_dtype=dtype
        ).to(device)
    for module in (text, multi_adapter, teacher, student, vae, hybrid_adapter, hybrid):
        if module is None:
            continue
        module.requires_grad_(False)
    return {
        "text": text,
        "multi_adapter": multi_adapter,
        "hybrid_adapter": hybrid_adapter,
        "teacher": teacher,
        "student": student,
        "hybrid": hybrid,
        "vae": vae,
        "first_frame": first_frame,
        "scheduler": PyramidFlowMatchEulerDiscreteScheduler(),
        "modifier": DEFAULT_PROMPT_MODIFIER,
        "negative": DEFAULT_NEGATIVE_PROMPT,
        "checkpoint_step": checkpoint_step,
        "model_path": str(local_model_path),
    }


@torch.inference_mode()
def evaluate_prompt(
    *,
    models: dict[str, object],
    prompt: str,
    args: argparse.Namespace,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> dict[str, object]:
    prompt_with_modifier = prompt + str(models["modifier"])
    teacher_first = make_condition(
        text=models["text"], adapter=models["multi_adapter"], prompt=prompt_with_modifier,
        negative_prompt=str(models["negative"]), guidance=7.0, device=device,
    )
    teacher_video = make_condition(
        text=models["text"], adapter=models["multi_adapter"], prompt=prompt_with_modifier,
        negative_prompt=str(models["negative"]), guidance=5.0, device=device,
    )
    dmd_condition = make_condition(
        text=models["text"], adapter=models["multi_adapter"], prompt=prompt_with_modifier,
        negative_prompt=str(models["negative"]), guidance=0.0, device=device,
    )
    hybrid_condition = None
    if models["hybrid"] is not None and models["hybrid_adapter"] is not None:
        hybrid_condition = make_condition(
            text=models["text"], adapter=models["hybrid_adapter"], prompt=prompt_with_modifier,
            negative_prompt=str(models["negative"]), guidance=0.0, device=device,
        )

    generator = torch.Generator(device=device).manual_seed(args.seed + 100_000 * index)
    full_noise = torch.randn(
        1, 16, ((args.num_frames - 1) // 8) + 1, args.height // 8, args.width // 8,
        device=device, dtype=dtype, generator=generator,
    )
    scheduler = models["scheduler"]
    transition_seed = args.seed + 1_000_000 * (index + 1)
    teacher = run_rollout(
        dit=models["teacher"], scheduler=scheduler, full_noise=full_noise,
        condition_for_unit=lambda unit: teacher_first if unit == 0 else teacher_video,
        first_steps=args.first_steps, video_steps=args.video_steps,
        transition_seed=transition_seed, initial_anchor=None,
    )
    dmd_teacher_forced = run_rollout(
        dit=models["student"], scheduler=scheduler, full_noise=full_noise,
        condition_for_unit=lambda _: dmd_condition,
        first_steps=1, video_steps=1, transition_seed=transition_seed,
        initial_anchor=teacher.latents[:, :, :1], teacher_history=teacher.latents,
    )
    dmd_teacher_anchor = run_rollout(
        dit=models["student"], scheduler=scheduler, full_noise=full_noise,
        condition_for_unit=lambda _: dmd_condition,
        first_steps=1, video_steps=1, transition_seed=transition_seed,
        initial_anchor=teacher.latents[:, :, :1],
    )
    systems = {
        "teacher_multistep": teacher,
        "dmd_teacher_forced": dmd_teacher_forced,
        "dmd_teacher_anchor_self_history": dmd_teacher_anchor,
    }
    ssd_anchor = None
    anchor_image = None
    if models["first_frame"] is not None:
        ssd_anchor, anchor_image = encode_ssd_anchor(
            pipeline=models["first_frame"], vae=models["vae"], prompt=prompt_with_modifier,
            height=args.height, width=args.width, seed=args.seed + 10_000 * index,
            device=device, dtype=dtype,
        )
        systems["dmd_ssd_anchor_self_history"] = run_rollout(
            dit=models["student"], scheduler=scheduler, full_noise=full_noise,
            condition_for_unit=lambda _: dmd_condition,
            first_steps=1, video_steps=1, transition_seed=transition_seed,
            initial_anchor=ssd_anchor,
        )
    if hybrid_condition is not None and ssd_anchor is not None:
        systems["released_hybrid_ssd_anchor"] = run_rollout(
            dit=models["hybrid"], scheduler=scheduler, full_noise=full_noise,
            condition_for_unit=lambda _: hybrid_condition,
            first_steps=1, video_steps=1, transition_seed=transition_seed,
            initial_anchor=ssd_anchor,
        )
    frames = {name: decode_latents(models["vae"], result.latents) for name, result in systems.items()}
    prompt_dir = output_dir / f"prompt_{index:02d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    if anchor_image is not None:
        anchor_image.save(prompt_dir / "ssd1b_anchor.png")
    for name, values in frames.items():
        export_to_video([Image.fromarray(frame) for frame in values], prompt_dir / f"{name}.mp4", fps=args.fps)
    torch.save({name: result.latents.cpu() for name, result in systems.items()}, prompt_dir / "rollout_latents.pt")

    rollout_relative_to_teacher = {
        name: tensor_metrics(teacher.latents[:, :, 1:], result.latents[:, :, 1:])
        for name, result in systems.items() if name != "teacher_multistep"
    }
    return {
        "prompt": prompt,
        "teacher_target_mismatch": teacher_target_mismatch(teacher),
        "oracle_local_dmd": oracle_local_dmd_metrics(
            teacher=teacher, student=models["student"], condition=dmd_condition, scheduler=scheduler
        ),
        "anchor_latent_difference": (
            None if ssd_anchor is None else tensor_metrics(teacher.latents[:, :, :1], ssd_anchor)
        ),
        "rollout_relative_to_teacher": rollout_relative_to_teacher,
        "systems": {
            name: {
                "latent_motion": latent_motion_metrics(result.latents),
                "decoded_motion": decoded_motion_metrics(frames[name]),
            }
            for name, result in systems.items()
        },
        "files": {name: str((prompt_dir / f"{name}.mp4").resolve()) for name in systems},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--first-steps", type=int, default=20)
    parser.add_argument("--video-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Also load the released Hybrid DiT control; disabled by default to bound local audit memory.",
    )
    parser.add_argument(
        "--include-ssd-anchor",
        action="store_true",
        help="Load SSD1B and add the deployed-anchor branch; disabled by default to fit the local audit cgroup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The DMD motion audit requires a CUDA GPU.")
    if args.num_frames != 49 or args.height != 320 or args.width != 512:
        raise ValueError("This audit is pinned to NeoDragon's released 320x512, 49-frame protocol.")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or list(DEFAULT_PROMPTS)

    started = time.perf_counter()
    models = load_models(
        load_config(args.config),
        checkpoint,
        device,
        dtype,
        include_hybrid=args.include_hybrid,
        include_ssd_anchor=args.include_ssd_anchor,
    )
    torch.cuda.synchronize(device)
    report: dict[str, object] = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(models["checkpoint_step"]),
            "teacher": "released 320p multistep DiT, 20 first-unit and 10 video-unit calls per stage, CFG 7/5",
            "student": "10k DMD DiT, 1-1-1 conditional rollout",
            "controls": [
                "DMD with teacher anchor and teacher-forced prior units",
                "DMD with teacher anchor and deployed self-history",
                "DMD with SSD1B anchor and deployed self-history (optional)",
                "released Hybrid with the same SSD1B anchor and self-history (optional)",
            ],
            "stage_target_test": "true multistep teacher stage endpoints versus bilinear pyramid targets used by current trainer",
            "noise_control": "same full latent noise and deterministic per-unit/stage corrective noise across every rollout",
        },
        "load_seconds": time.perf_counter() - started,
        "checkpoint_update": checkpoint_update_metrics(models["teacher"], models["student"]),
        "prompts": [],
    }
    for index, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        prompt_started = time.perf_counter()
        result = evaluate_prompt(
            models=models, prompt=prompt, args=args, index=index,
            device=device, dtype=dtype, output_dir=output_dir,
        )
        result["seconds"] = time.perf_counter() - prompt_started
        report["prompts"].append(result)
        print(
            f"Completed prompt {index + 1}/{len(prompts)} in {result['seconds']:.1f}s: {prompt}",
            flush=True,
        )

    report["total_seconds"] = time.perf_counter() - started
    (output_dir / "motion_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["protocol"], indent=2), flush=True)
    print(json.dumps(report["checkpoint_update"], indent=2), flush=True)
    print(f"Saved complete motion audit to {output_dir / 'motion_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
