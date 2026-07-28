#!/usr/bin/env python
"""Measure single-GPU memory for backpropagating through NeoDragon rollout calls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config

GIB = 1024**3


def dtype_from_name(name: str) -> torch.dtype:
    if str(name).lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if str(name).lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def cuda_memory(device: torch.device) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "device_free_gib": free_bytes / GIB,
        "device_total_gib": total_bytes / GIB,
    }


def pyramid_latents(x: torch.Tensor, num_stages: int) -> list[torch.Tensor]:
    """Differentiable replacement for NeoDragon's no-grad helper."""
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
    generated: list[torch.Tensor],
    num_stages: int,
) -> list[list[torch.Tensor]]:
    """Build causal history while retaining gradients to generated units."""
    if not generated:
        return [[] for _ in range(num_stages)]

    frames_per_unit = generated[0].shape[2]
    unit_index = len(generated)
    history_pyramid = pyramid_latents(torch.cat(generated, dim=2), num_stages)
    outputs: list[list[torch.Tensor]] = []
    for stage in range(num_stages):
        stage_input = [history_pyramid[stage][:, :, -frames_per_unit:]]
        current_unit = 1
        current_stage = stage
        while current_unit < unit_index:
            current_stage = max(current_stage - 1, 0)
            if current_stage == 0:
                break
            current_unit += 1
            begin = -(current_unit * frames_per_unit)
            end = -((current_unit - 1) * frames_per_unit)
            stage_input.append(history_pyramid[current_stage][:, :, begin:end])
        if current_stage == 0 and current_unit < unit_index:
            stage_input.append(
                history_pyramid[0][:, :, : -(current_unit * frames_per_unit)]
            )
        outputs.append(list(reversed(stage_input)))
    return outputs


def upsample_pyramidal_latent(
    latents: torch.Tensor,
    *,
    orig_sigma: float,
    gamma: float,
) -> torch.Tensor:
    """NeoDragon's differentiable coarse-to-fine transition."""
    temporal = latents.shape[2]
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    latents = F.interpolate(latents, scale_factor=2, mode="nearest")
    latents = rearrange(latents, "(b t) c h w -> b c t h w", t=temporal)

    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_sigma) + orig_sigma)
    beta = alpha * (1 - orig_sigma) / math.sqrt(gamma)
    return alpha * latents + beta * torch.randn_like(latents)


def load_bridge(
    cfg,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> MobileOVNeodragonTextBridge:
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload.get("student_state", payload))
    bridge.load_state_dict(state, strict=True)
    bridge.train()
    return bridge


def load_dit(cfg, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import DIT_ID
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler

    dit = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{DIT_ID}",
        torch_dtype=dtype,
    ).to(device)
    dit.eval().requires_grad_(False)
    return dit, PyramidFlowMatchEulerDiscreteScheduler(), Path(local_model_path)


def load_native_text_teacher(
    local_model_path: Path,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load and retain the native text teacher, matching Exp1 residency."""
    from neodragon import CONTEXT_ADAPTER_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.text_encoder_bundle import TextEncoderBundle

    teacher = TextEncoderBundle.from_pretrained(
        str(local_model_path),
        torch_dtype=dtype,
    ).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    teacher.requires_grad_(False)
    context_adapter.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        tokens, mask, pooled = teacher([prompt], device)
        tokens = context_adapter(tokens)
    return teacher, context_adapter, tokens, mask, pooled


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    cfg = load_config(args.config)
    device = torch.device("cuda", 0)
    dtype = dtype_from_name(args.dtype or cfg.backend.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dit, scheduler, local_model_path = load_dit(cfg, device, dtype)
    bridge = load_bridge(cfg, Path(args.bridge_checkpoint), device, dtype)
    native_teacher = None
    native_context_adapter = None
    native_tokens = None
    native_mask = None
    native_pooled = None
    if args.retain_native_text_teacher:
        (
            native_teacher,
            native_context_adapter,
            native_tokens,
            native_mask,
            native_pooled,
        ) = load_native_text_teacher(
            local_model_path,
            args.prompt,
            device,
            dtype,
        )
    trainable = [parameter for parameter in bridge.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.AdamW(trainable, lr=1e-5, foreach=False)
        if args.optimizer_step
        else None
    )

    prompt = args.prompt
    with torch.autocast("cuda", dtype=dtype):
        prompt_tokens, prompt_mask, pooled = bridge([prompt])

    num_stages = 3
    num_units = 6
    total_calls = num_units * num_stages
    if args.window_calls < 1 or args.window_calls > total_calls:
        raise ValueError(f"--window-calls must be in [1, {total_calls}]")
    first_grad_call = total_calls - args.window_calls

    # The hybrid pipeline starts from a one-frame 320x512 VAE anchor.
    generated = [
        torch.randn(1, 16, 1, 40, 64, device=device, dtype=dtype)
    ]
    tracked_losses: list[torch.Tensor] = []
    call_index = 0
    forward_start = None
    torch.cuda.synchronize(device)
    base_memory = cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for _unit in range(num_units):
            # History is rebuilt per autoregressive unit, matching NeoDragon inference.
            past = prepare_past_conditions(generated, num_stages)
            current = torch.randn(1, 16, 1, 10, 16, device=device, dtype=dtype)
            for stage in range(num_stages):
                tracking = call_index >= first_grad_call
                if tracking and forward_start is None:
                    generated = [value.detach() for value in generated]
                    current = current.detach()
                    past = prepare_past_conditions(generated, num_stages)
                    forward_start = time.perf_counter()

                if stage > 0:
                    context = nullcontext() if tracking else torch.no_grad()
                    with context:
                        current = upsample_pyramidal_latent(
                            current,
                            orig_sigma=1 - scheduler.orig_start_sigmas[stage],
                            gamma=scheduler.config.gamma,
                        )

                timesteps = scheduler.get_stage_timesteps(1, stage, device=device)
                sigmas = scheduler.get_stage_sigmas(1, stage, device=device)
                timestep = timesteps[0].expand(1).to(dtype)
                sigma = sigmas[0].to(dtype)
                sigma_next = sigmas[1].to(dtype)
                stage_input = past[stage] + [current]

                if not tracking:
                    dit.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        prediction = dit(
                            sample=[stage_input],
                            encoder_hidden_states=prompt_tokens.detach(),
                            encoder_attention_mask=prompt_mask,
                            pooled_projections=pooled.detach(),
                            timestep_ratio=timestep,
                        )[0]
                        current = scheduler.step(
                            model_output=prediction,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            sample=current,
                        ).prev_sample
                else:
                    # A no-grad teacher call is included because rollout distillation
                    # must retain earlier student activations while querying a teacher.
                    dit.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        teacher_prediction = dit(
                            sample=[[value.detach() for value in stage_input]],
                            encoder_hidden_states=(
                                native_tokens
                                if native_tokens is not None
                                else prompt_tokens.detach()
                            ),
                            encoder_attention_mask=(
                                native_mask if native_mask is not None else prompt_mask
                            ),
                            pooled_projections=(
                                native_pooled
                                if native_pooled is not None
                                else pooled.detach()
                            ),
                            timestep_ratio=timestep,
                        )[0]

                    dit.train(args.gradient_checkpointing)
                    dit.gradient_checkpointing = bool(args.gradient_checkpointing)
                    dit.gradient_checkpointing_ratio = float(args.checkpoint_ratio)
                    with torch.autocast("cuda", dtype=dtype):
                        student_prediction = dit(
                            sample=[stage_input],
                            encoder_hidden_states=prompt_tokens,
                            encoder_attention_mask=prompt_mask,
                            pooled_projections=pooled,
                            timestep_ratio=timestep,
                        )[0]
                        # Scaling the detached target avoids an exactly-zero gradient
                        # while preserving the teacher call's realistic tensor shape.
                        target = teacher_prediction.detach() * 0.97
                        tracked_losses.append(
                            F.mse_loss(
                                student_prediction.float(),
                                target.float(),
                            )
                        )
                        current = scheduler.step(
                            model_output=student_prediction,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            sample=current,
                        ).prev_sample
                call_index += 1
            generated.append(current)

        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - float(forward_start)
        after_forward = cuda_memory(device)
        loss = torch.stack(tracked_losses).mean()

        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_start
        after_backward = cuda_memory(device)
        grad_norm_sq = sum(
            float(parameter.grad.float().square().sum())
            for parameter in trainable
            if parameter.grad is not None
        )

        optimizer_seconds = 0.0
        if optimizer is not None:
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_start
        after_optimizer = cuda_memory(device)

        result = {
            "status": "ok",
            "gpu": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "resolution": [320, 512],
            "pixel_frames": 49,
            "latent_anchor_shape": [1, 16, 1, 40, 64],
            "rollout_calls_total": total_calls,
            "window_calls": args.window_calls,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "checkpoint_ratio": (
                float(args.checkpoint_ratio) if args.gradient_checkpointing else None
            ),
            "optimizer_step": bool(args.optimizer_step),
            "retain_native_text_teacher": bool(args.retain_native_text_teacher),
            "trainable_bridge_parameters": sum(p.numel() for p in trainable),
            "bridge_grad_norm": math.sqrt(grad_norm_sq),
            "loss": float(loss.detach()),
            "base_memory": base_memory,
            "after_forward": after_forward,
            "after_backward": after_backward,
            "after_optimizer": after_optimizer,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
        }
    except torch.OutOfMemoryError as exc:
        result = {
            "status": "oom",
            "gpu": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "window_calls": args.window_calls,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "checkpoint_ratio": (
                float(args.checkpoint_ratio) if args.gradient_checkpointing else None
            ),
            "optimizer_step": bool(args.optimizer_step),
            "retain_native_text_teacher": bool(args.retain_native_text_teacher),
            "base_memory": base_memory,
            "at_oom": cuda_memory(device),
            "error": str(exc),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    del optimizer, bridge, dit, native_teacher, native_context_adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--bridge-checkpoint",
        default=(
            "checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/"
            "neodragon_text_bridge_latest.pt"
        ),
    )
    parser.add_argument("--window-calls", type=int, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-ratio", type=float, default=0.0)
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument(
        "--retain-native-text-teacher",
        action="store_true",
        help="Keep NeoDragon's native CLIP/T5 bundle and context adapter on GPU.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--prompt",
        default=(
            "A red fox walking through gentle snowfall, cinematic wildlife footage, "
            "realistic textures, high detail, natural colours"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
