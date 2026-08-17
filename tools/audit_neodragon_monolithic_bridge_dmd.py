#!/usr/bin/env python
"""Audit a monolithic text bridge against an all-native-unit NeoDragon DMD student.

The bridge and the DMD student are evaluated independently of image anchors:
every run starts from the native text-to-video seven-unit noise trajectory.
For each prompt, the audit measures the direct post-context contract, all 21
one-step DiT flow responses on the same native state trajectory, and the
resulting full DMD latent-rollout drift under identical random seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramidal_dmd import DMDCondition, predict_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--dmd-checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def require_slurm_cuda() -> None:
    import os

    if not torch.cuda.is_available():
        raise RuntimeError("This compatibility audit requires CUDA.")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Run CUDA compatibility audits through srun or sbatch, never directly.")


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def tensor_similarity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Tensor shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    if reference_mask is not None or candidate_mask is not None:
        if reference_mask is None or candidate_mask is None:
            raise ValueError("Both masks are required for masked token comparison.")
        mask = (reference_mask.bool() & candidate_mask.bool()).unsqueeze(-1)
        reference = reference.masked_select(mask).reshape(1, -1)
        candidate = candidate.masked_select(mask).reshape(1, -1)
    else:
        reference = reference.reshape(1, -1)
        candidate = candidate.reshape(1, -1)
    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    return {
        "cosine": float(F.cosine_similarity(reference, candidate, dim=-1).mean().cpu()),
        "relative_l2": float((difference.norm() / reference.norm().clamp_min(1e-12)).cpu()),
        "relative_mse": float(
            (difference.square().mean() / reference.square().mean().clamp_min(1e-12)).cpu()
        ),
    }


def reset_rng(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def stage_endpoints(intermediates: list[torch.Tensor], steps: list[int]) -> list[torch.Tensor]:
    if len(intermediates) != sum(steps):
        raise RuntimeError(f"Expected {sum(steps)} stage states, got {len(intermediates)}")
    result: list[torch.Tensor] = []
    offset = 0
    for count in steps:
        offset += int(count)
        result.append(intermediates[offset - 1].detach().cpu())
    return result


@torch.inference_mode()
def explicit_one_step_rollout(
    *,
    dit: torch.nn.Module,
    scheduler: Any,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    pooled: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Replay the released seven-unit 1-1-1 sampler with an explicit condition."""

    import neodragon.utils.generation_utils as generation

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    reset_rng(args.seed)
    latents = generation._prepare_latent_noise(
        1,
        int(dit.config.in_channels),
        7,
        args.height // 8,
        args.width // 8,
        dtype,
        device,
    )
    latents = generation._downsample_noise_2x(latents, 2)
    generated: list[torch.Tensor] = []
    captured: list[list[torch.Tensor]] = []
    one_step = [1, 1, 1]
    for unit in range(7):
        history = generation._prepare_past_condition_latents(generated, 3, False)
        final, intermediates = generation._generate_one_unit(
            scheduler=scheduler,
            dit=dit,
            num_stages=3,
            latents=latents[:, :, unit : unit + 1],
            past_conditions=history,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            num_inference_steps=one_step,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=False,
            guidance_scale=7.0,
            video_guidance_scale=5.0,
            show_denoising=True,
        )
        generated.append(final)
        captured.append(stage_endpoints(intermediates, one_step))
    return torch.cat(generated, dim=2).detach().cpu(), captured


def resolve_models(
    cfg,
    *,
    dmd_checkpoint: Path,
    bridge_checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
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
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    dmd_payload = torch.load(dmd_checkpoint, map_location="cpu", weights_only=False)
    if dmd_payload.get("schedule") != "pyramidal_1-1-1_all_native_units":
        raise ValueError(
            "This audit only accepts the corrected DMD-v2 all-native-unit schedule; "
            f"received {dmd_payload.get('schedule')!r}."
        )
    if dmd_payload.get("context_adapter_id") != MULTISTEP_CONTEXT_ADAPTER_ID:
        raise ValueError("DMD checkpoint does not use the released multi-step context adapter.")
    if dmd_payload.get("teacher_dit_id") != MULTISTEP_DIT_ID:
        raise ValueError("DMD checkpoint does not use the released multi-step DiT base.")

    bridge_payload = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    bridge_stack = bridge_payload.get("teacher_stack")
    if not isinstance(bridge_stack, dict) or bridge_stack.get("name") != "multistep":
        raise ValueError(
            "Bridge checkpoint is not explicitly marked as a multistep target. "
            "Train it with scripts/train_neodragon_monolithic_text_bridge_1node8gpu.sbatch."
        )
    if bridge_stack.get("context_adapter_id") != MULTISTEP_CONTEXT_ADAPTER_ID:
        raise ValueError("Bridge context-adapter metadata does not match the DMD-v2 contract.")
    if bridge_stack.get("dit_id") != MULTISTEP_DIT_ID:
        raise ValueError("Bridge DiT metadata does not match the DMD-v2 contract.")

    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}", torch_dtype=dtype
    ).to(device).eval()
    dmd = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
    ).to(device).eval()
    dmd_state = dmd_payload.get("student")
    if not isinstance(dmd_state, dict):
        raise ValueError(f"DMD checkpoint does not contain a student state: {dmd_checkpoint}")
    dmd.load_state_dict(dmd_state, strict=True)

    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    bridge_state = bridge_payload.get("bridge")
    if not isinstance(bridge_state, dict):
        raise ValueError(f"Bridge checkpoint does not contain a bridge state: {bridge_checkpoint}")
    bridge.load_state_dict(bridge_state, strict=True)
    for module in (text, adapter, dmd, bridge):
        module.requires_grad_(False)
    return {
        "text": text,
        "adapter": adapter,
        "dmd": dmd,
        "bridge": bridge,
        "scheduler": PyramidFlowMatchEulerDiscreteScheduler(),
        "prompt_modifier": DEFAULT_PROMPT_MODIFIER,
        "dmd_payload": dmd_payload,
        "bridge_payload": bridge_payload,
    }


@torch.inference_mode()
def encode_conditions(
    *, models: dict[str, Any], prompt: str, device: torch.device
) -> tuple[DMDCondition, DMDCondition, dict[str, dict[str, float]]]:
    full_prompt = prompt + models["prompt_modifier"]
    native_tokens, native_mask, native_pooled = models["text"]([full_prompt], device)
    native_tokens = models["adapter"](native_tokens)
    bridge_tokens, bridge_mask, bridge_pooled = models["bridge"]([full_prompt])
    condition_metrics = {
        "tokens": tensor_similarity(
            native_tokens,
            bridge_tokens,
            reference_mask=native_mask,
            candidate_mask=bridge_mask,
        ),
        "pooled": tensor_similarity(native_pooled, bridge_pooled),
        "mask_agreement": {
            "fraction": float((native_mask.bool() == bridge_mask.bool()).float().mean().cpu())
        },
    }
    return (
        DMDCondition(tokens=native_tokens, mask=native_mask, pooled=native_pooled),
        DMDCondition(tokens=bridge_tokens, mask=bridge_mask, pooled=bridge_pooled),
        condition_metrics,
    )


@torch.inference_mode()
def flow_response_audit(
    *,
    models: dict[str, Any],
    native_condition: DMDCondition,
    bridge_condition: DMDCondition,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compare conditions before each of the 21 deployed DMD calls."""

    import neodragon.utils.generation_utils as generation

    scheduler = models["scheduler"]
    reset_rng(seed)
    full_noise = generation._prepare_latent_noise(
        1,
        int(models["dmd"].config.in_channels),
        7,
        args.height // 8,
        args.width // 8,
        dtype,
        device,
    )
    low_noise = generation._downsample_noise_2x(full_noise, 2)
    generated: list[torch.Tensor] = []
    response_metrics: list[dict[str, float]] = []
    by_call: list[dict[str, object]] = []
    for unit in range(7):
        histories = generation._prepare_past_condition_latents(generated, 3, False)
        current = low_noise[:, :, unit : unit + 1]
        for stage in range(3):
            if stage > 0:
                current = generation._upsample_pyramidal_latent(
                    latents=current,
                    orig_sigma=1.0 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    device=device,
                    dtype=dtype,
                )
            timestep = scheduler.get_stage_timesteps(1, stage, device=device)[0]
            history = tuple(histories[stage])
            native_flow = predict_flow(
                dit=models["dmd"],
                current=current,
                history=history,
                condition=native_condition,
                timestep=timestep,
            )
            bridge_flow = predict_flow(
                dit=models["dmd"],
                current=current,
                history=history,
                condition=bridge_condition,
                timestep=timestep,
            )
            metric = tensor_similarity(native_flow, bridge_flow)
            response_metrics.append(metric)
            by_call.append({"unit": unit, "stage": stage, **metric})
            sigmas = scheduler.get_stage_sigmas(1, stage, device=device)
            current = scheduler.step(
                model_output=native_flow,
                sigma=sigmas[0].to(dtype=current.dtype),
                sigma_next=sigmas[1].to(dtype=current.dtype),
                sample=current,
            ).prev_sample
        generated.append(current)
    return average(response_metrics), by_call


@torch.inference_mode()
def rollout_drift_audit(
    *,
    models: dict[str, Any],
    native_condition: DMDCondition,
    bridge_condition: DMDCondition,
    args: argparse.Namespace,
) -> dict[str, object]:
    native_latents, native_stages = explicit_one_step_rollout(
        dit=models["dmd"],
        scheduler=models["scheduler"],
        prompt_embeds=native_condition.tokens,
        prompt_mask=native_condition.mask,
        pooled=native_condition.pooled,
        args=args,
    )
    bridge_latents, bridge_stages = explicit_one_step_rollout(
        dit=models["dmd"],
        scheduler=models["scheduler"],
        prompt_embeds=bridge_condition.tokens,
        prompt_mask=bridge_condition.mask,
        pooled=bridge_condition.pooled,
        args=args,
    )
    stage_metrics: dict[str, dict[str, float]] = {}
    for unit, (native_unit, bridge_unit) in enumerate(zip(native_stages, bridge_stages)):
        for stage, (native_stage, bridge_stage) in enumerate(zip(native_unit, bridge_unit)):
            stage_metrics[f"unit{unit}_stage{stage}"] = tensor_similarity(native_stage, bridge_stage)
    return {
        "full_latent": tensor_similarity(native_latents, bridge_latents),
        "native_motion": latent_motion_metrics(native_latents),
        "bridge_motion": latent_motion_metrics(bridge_latents),
        "stage_endpoints": stage_metrics,
    }


@torch.inference_mode()
def audit_prompt(
    *, models: dict[str, Any], prompt: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype, seed: int
) -> dict[str, object]:
    native_condition, bridge_condition, condition_metrics = encode_conditions(
        models=models, prompt=prompt, device=device
    )
    flow_metrics, flow_by_call = flow_response_audit(
        models=models,
        native_condition=native_condition,
        bridge_condition=bridge_condition,
        args=args,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    rollout_metrics = rollout_drift_audit(
        models=models,
        native_condition=native_condition,
        bridge_condition=bridge_condition,
        args=args,
    )
    return {
        "prompt": prompt,
        "condition": condition_metrics,
        "flow_response": flow_metrics,
        "flow_by_call": flow_by_call,
        "rollout": rollout_metrics,
    }


def main() -> None:
    args = parse_args()
    require_slurm_cuda()
    prompts = args.prompt or [
        "An astronaut walks slowly across the red surface of Mars as dust blows behind them.",
        "A vintage red car drives along a coastal road at golden hour, the camera tracking beside it.",
        "A wide cinematic view of a waterfall flowing into a misty tropical valley, leaves moving in the wind.",
    ]
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    models = resolve_models(
        load_config(args.config),
        dmd_checkpoint=Path(args.dmd_checkpoint).resolve(),
        bridge_checkpoint=Path(args.bridge_checkpoint).resolve(),
        device=device,
        dtype=dtype,
    )
    report: dict[str, object] = {
        "protocol": {
            "state_protocol": "native T2V: seven units x three one-step Pyramidal stages, no image anchor",
            "condition_protocol": "released multistep post-context condition versus monolithic-trained bridge",
            "flow_protocol": "same native DMD state at every one of 21 calls, conditional student path",
            "rollout_protocol": "two complete DMD rollouts from the same seed",
            "dmd_checkpoint": str(Path(args.dmd_checkpoint).resolve()),
            "dmd_step": int(models["dmd_payload"].get("step", -1)),
            "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
            "bridge_step": int(models["bridge_payload"].get("step", -1)),
        },
        "prompts": [],
    }
    for index, prompt in enumerate(prompts):
        report["prompts"].append(
            audit_prompt(
                models=models,
                prompt=prompt,
                args=args,
                device=device,
                dtype=dtype,
                seed=args.seed + index * 1009,
            )
        )
    prompt_reports = report["prompts"]
    assert isinstance(prompt_reports, list)
    report["summary"] = {
        "condition_tokens": average([item["condition"]["tokens"] for item in prompt_reports]),
        "condition_pooled": average([item["condition"]["pooled"] for item in prompt_reports]),
        "flow_response": average([item["flow_response"] for item in prompt_reports]),
        "rollout_full_latent": average([item["rollout"]["full_latent"] for item in prompt_reports]),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "monolithic_bridge_dmd_compatibility.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["protocol"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved compatibility report: {output_path}")


if __name__ == "__main__":
    main()
