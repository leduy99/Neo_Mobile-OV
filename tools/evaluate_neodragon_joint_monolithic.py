#!/usr/bin/env python
"""Compare jointly fine-tuned monolithic NeoDragon checkpoints under fixed noise."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramidal_dmd import DMDCondition
from tools import audit_neodragon_dmd_motion as motion
from tools.audit_neodragon_monolithic_bridge_dmd import tensor_similarity
from tools.evaluate_neodragon_monolithic_bridge_dmd import decoded_pair_metrics


DEFAULT_PROMPTS = (
    "An astronaut walks slowly across the red surface of Mars as dust blows behind them.",
    "A vintage red car drives along a coastal road at golden hour, the camera tracking beside it.",
    "A wide cinematic view of a waterfall flowing into a misty tropical valley, leaves moving in the wind.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeated LABEL=/path/to/neodragon_dit_bridge_latest.pt specification.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--external-anchor",
        action="store_true",
        help=(
            "Generate unit zero once with the released native stack, then reuse that exact "
            "latent anchor while evaluating video units 1..6."
        ),
    )
    return parser.parse_args()


def require_slurm_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires a CUDA allocation.")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Run this evaluator through srun or sbatch.")


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be LABEL=PATH, received {value!r}")
        label, raw_path = value.split("=", 1)
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip())
        path = Path(raw_path).expanduser().resolve()
        if not label or label in labels:
            raise ValueError(f"Checkpoint label must be unique and non-empty: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        result.append((label, path))
    return result


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def cpu_condition(condition: DMDCondition) -> DMDCondition:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach().cpu()

    return DMDCondition(
        tokens=move(condition.tokens),
        mask=move(condition.mask),
        pooled=move(condition.pooled),
        negative_tokens=move(condition.negative_tokens),
        negative_mask=move(condition.negative_mask),
        negative_pooled=move(condition.negative_pooled),
        guidance_scale=condition.guidance_scale,
    )


def cfg_condition(
    positive: DMDCondition,
    negative: DMDCondition,
    *,
    guidance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> DMDCondition:
    return DMDCondition(
        tokens=positive.tokens.to(device=device, dtype=dtype),
        mask=positive.mask.to(device=device),
        pooled=positive.pooled.to(device=device, dtype=dtype),
        negative_tokens=negative.tokens.to(device=device, dtype=dtype),
        negative_mask=negative.mask.to(device=device),
        negative_pooled=negative.pooled.to(device=device, dtype=dtype),
        guidance_scale=float(guidance),
    )


def condition_metrics(
    native_positive: DMDCondition,
    native_negative: DMDCondition,
    bridge_positive: DMDCondition,
    bridge_negative: DMDCondition,
) -> dict[str, Any]:
    return {
        "positive_tokens": tensor_similarity(
            native_positive.tokens,
            bridge_positive.tokens,
            reference_mask=native_positive.mask,
            candidate_mask=bridge_positive.mask,
        ),
        "positive_pooled": tensor_similarity(native_positive.pooled, bridge_positive.pooled),
        "negative_tokens": tensor_similarity(
            native_negative.tokens,
            bridge_negative.tokens,
            reference_mask=native_negative.mask,
            candidate_mask=bridge_negative.mask,
        ),
        "negative_pooled": tensor_similarity(native_negative.pooled, bridge_negative.pooled),
        "cfg_token_delta": tensor_similarity(
            native_positive.tokens - native_negative.tokens,
            bridge_positive.tokens - bridge_negative.tokens,
            reference_mask=native_positive.mask,
            candidate_mask=bridge_positive.mask,
        ),
        "cfg_pooled_delta": tensor_similarity(
            native_positive.pooled - native_negative.pooled,
            bridge_positive.pooled - bridge_negative.pooled,
        ),
    }


def encode_native_conditions(
    *,
    text: torch.nn.Module,
    adapter: torch.nn.Module,
    prompts: list[str],
    negative_prompt: str,
    modifier: str,
    device: torch.device,
) -> tuple[list[DMDCondition], DMDCondition]:
    negative_tokens, negative_mask, negative_pooled = text([negative_prompt], device)
    negative = cpu_condition(
        DMDCondition(
            tokens=adapter(negative_tokens),
            mask=negative_mask,
            pooled=negative_pooled,
        )
    )
    positives: list[DMDCondition] = []
    for prompt in prompts:
        tokens, mask, pooled = text([prompt + modifier], device)
        positives.append(
            cpu_condition(DMDCondition(tokens=adapter(tokens), mask=mask, pooled=pooled))
        )
    return positives, negative


def encode_bridge_conditions(
    *,
    bridge: torch.nn.Module,
    prompts: list[str],
    negative_prompt: str,
    modifier: str,
) -> tuple[list[DMDCondition], DMDCondition]:
    negative_tokens, negative_mask, negative_pooled = bridge([negative_prompt])
    negative = cpu_condition(
        DMDCondition(tokens=negative_tokens, mask=negative_mask, pooled=negative_pooled)
    )
    positives: list[DMDCondition] = []
    for prompt in prompts:
        tokens, mask, pooled = bridge([prompt + modifier])
        positives.append(cpu_condition(DMDCondition(tokens=tokens, mask=mask, pooled=pooled)))
    return positives, negative


@torch.inference_mode()
def run_cfg_rollout(
    *,
    dit: torch.nn.Module,
    scheduler: Any,
    positive: DMDCondition,
    negative: DMDCondition,
    full_noise: torch.Tensor,
    transition_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_anchor: torch.Tensor | None = None,
) -> tuple[motion.Rollout, float]:
    first = cfg_condition(positive, negative, guidance=7.0, device=device, dtype=dtype)
    video = cfg_condition(positive, negative, guidance=5.0, device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    rollout = motion.run_rollout(
        dit=dit,
        scheduler=scheduler,
        full_noise=full_noise,
        condition_for_unit=lambda unit: first if unit == 0 else video,
        first_steps=20,
        video_steps=10,
        transition_seed=transition_seed,
        initial_anchor=initial_anchor,
    )
    torch.cuda.synchronize()
    return rollout, time.perf_counter() - start


def prompt_noise(
    *, args: argparse.Namespace, index: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed + index * 1009)
    return torch.randn(
        1,
        channels,
        ((args.num_frames - 1) // 8) + 1,
        args.height // 8,
        args.width // 8,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def save_system(
    *,
    name: str,
    rollout: motion.Rollout,
    seconds: float,
    vae: torch.nn.Module,
    prompt_dir: Path,
    fps: int,
) -> tuple[dict[str, Any], torch.Tensor, np.ndarray]:
    frames = motion.decode_latents(vae, rollout.latents)
    video_path = prompt_dir / f"{name}.mp4"
    export_to_video([Image.fromarray(frame) for frame in frames], video_path, fps=fps)
    latents = rollout.latents.detach().cpu()
    return (
        {
            "inference_seconds": float(seconds),
            "latent_motion": motion.latent_motion_metrics(latents),
            "decoded_motion": motion.decoded_motion_metrics(frames),
            "video": str(video_path.resolve()),
        },
        latents,
        frames,
    )


def make_contact_sheet(path: Path, systems: dict[str, np.ndarray]) -> None:
    frame_indices = (0, 24, 48)
    tile_width, tile_height, header = 256, 160, 28
    canvas = Image.new(
        "RGB", (len(frame_indices) * tile_width, len(systems) * (tile_height + header)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, (name, frames) in enumerate(systems.items()):
        y = row * (tile_height + header)
        draw.text((4, y + 5), name, fill="black")
        for column, frame_index in enumerate(frame_indices):
            frame = Image.fromarray(frames[frame_index]).resize((tile_width, tile_height))
            canvas.paste(frame, (column * tile_width, y + header))
    canvas.save(path, quality=92)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    stack = payload.get("teacher_stack")
    if not isinstance(stack, dict) or stack.get("name") != "multistep":
        raise ValueError(f"Checkpoint is not a multistep joint checkpoint: {path}")
    if not isinstance(payload.get("dit"), dict) or not isinstance(payload.get("bridge"), dict):
        raise ValueError(f"Checkpoint must contain both dit and bridge state dicts: {path}")
    return payload


def main() -> None:
    args = parse_args()
    require_slurm_cuda()
    if (args.height, args.width, args.num_frames) != (320, 512, 49):
        raise ValueError("Evaluation is fixed to the native 320x512, 49-frame contract.")
    checkpoints = parse_checkpoints(args.checkpoint)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = motion.dtype_from_name(args.dtype)
    cfg = load_config(args.config)

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
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT_MODIFIER

    print("Encoding native positive and negative conditions...", flush=True)
    text = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}", torch_dtype=dtype
    ).to(device).eval()
    native_positives, native_negative = encode_native_conditions(
        text=text,
        adapter=adapter,
        prompts=prompts,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
    )
    del text, adapter
    gc.collect()
    torch.cuda.empty_cache()

    vae = AsymmetricCausalVideoVAE.from_pretrained(
        f"{local_model_path}/causal_video_vae", torch_dtype=dtype
    ).to(device).eval()
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    prompt_reports = [
        {"prompt": prompt, "systems": {}, "comparisons": {}, "conditions": {}}
        for prompt in prompts
    ]
    latent_cache: list[dict[str, torch.Tensor]] = [dict() for _ in prompts]
    frame_cache: list[dict[str, np.ndarray]] = [dict() for _ in prompts]
    anchor_cache: list[torch.Tensor | None] = [None for _ in prompts]

    print("Running released monolithic native-CFG baseline...", flush=True)
    native_dit = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
    ).to(device).eval()
    channels = int(native_dit.config.in_channels)
    for index, positive in enumerate(native_positives):
        prompt_dir = output_dir / f"prompt_{index:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        noise = prompt_noise(
            args=args, index=index, channels=channels, device=device, dtype=dtype
        )
        transition_seed = args.seed + 100_003 * (index + 1)
        initial_anchor = None
        if args.external_anchor:
            anchor_source, _ = run_cfg_rollout(
                dit=native_dit,
                scheduler=scheduler,
                positive=positive,
                negative=native_negative,
                full_noise=noise,
                transition_seed=transition_seed,
                device=device,
                dtype=dtype,
            )
            initial_anchor = anchor_source.latents[:, :, :1].detach()
            anchor_cache[index] = initial_anchor.cpu()
        rollout, seconds = run_cfg_rollout(
            dit=native_dit,
            scheduler=scheduler,
            positive=positive,
            negative=native_negative,
            full_noise=noise,
            transition_seed=transition_seed,
            device=device,
            dtype=dtype,
            initial_anchor=initial_anchor,
        )
        system, latents, frames = save_system(
            name="released_native_cfg",
            rollout=rollout,
            seconds=seconds,
            vae=vae,
            prompt_dir=prompt_dir,
            fps=args.fps,
        )
        prompt_reports[index]["systems"]["released_native_cfg"] = system
        latent_cache[index]["released_native_cfg"] = latents
        frame_cache[index]["released_native_cfg"] = frames
    del native_dit
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint_metadata: dict[str, Any] = {}
    for label, checkpoint_path in checkpoints:
        print(f"Loading and evaluating {label}: {checkpoint_path}", flush=True)
        payload = load_payload(checkpoint_path)
        checkpoint_metadata[label] = {
            "path": str(checkpoint_path),
            "step": int(payload.get("step", -1)),
            "bridge_initialization": payload.get("bridge_ckpt"),
            "objective": payload.get("objective"),
            "args": payload.get("args"),
        }
        bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
        bridge.load_state_dict(payload["bridge"], strict=True)
        bridge_positives, bridge_negative = encode_bridge_conditions(
            bridge=bridge,
            prompts=prompts,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            modifier=DEFAULT_PROMPT_MODIFIER,
        )
        del bridge
        gc.collect()
        torch.cuda.empty_cache()

        joint_dit = PyramidMMDiT.from_pretrained(
            f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
        ).to(device).eval()
        joint_dit.load_state_dict(payload["dit"], strict=True)
        del payload
        gc.collect()
        for index, (native_positive, bridge_positive) in enumerate(
            zip(native_positives, bridge_positives)
        ):
            prompt_dir = output_dir / f"prompt_{index:02d}"
            prompt_reports[index]["conditions"][label] = condition_metrics(
                native_positive,
                native_negative,
                bridge_positive,
                bridge_negative,
            )
            for condition_name, positive, negative in (
                ("native_text_cfg", native_positive, native_negative),
                ("bridge_cfg", bridge_positive, bridge_negative),
            ):
                system_name = f"{label}_{condition_name}"
                rollout, seconds = run_cfg_rollout(
                    dit=joint_dit,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    full_noise=prompt_noise(
                        args=args, index=index, channels=channels, device=device, dtype=dtype
                    ),
                    transition_seed=args.seed + 100_003 * (index + 1),
                    device=device,
                    dtype=dtype,
                    initial_anchor=(
                        None
                        if anchor_cache[index] is None
                        else anchor_cache[index].to(device=device, dtype=dtype)
                    ),
                )
                system, latents, frames = save_system(
                    name=system_name,
                    rollout=rollout,
                    seconds=seconds,
                    vae=vae,
                    prompt_dir=prompt_dir,
                    fps=args.fps,
                )
                prompt_reports[index]["systems"][system_name] = system
                latent_cache[index][system_name] = latents
                frame_cache[index][system_name] = frames
                prompt_reports[index]["comparisons"][f"{system_name}_vs_released"] = {
                    "latent": motion.tensor_metrics(
                        latent_cache[index]["released_native_cfg"], latents
                    ),
                    "decoded": decoded_pair_metrics(
                        frame_cache[index]["released_native_cfg"], frames
                    ),
                }
            native_name = f"{label}_native_text_cfg"
            bridge_name = f"{label}_bridge_cfg"
            prompt_reports[index]["comparisons"][f"{label}_bridge_vs_native_text"] = {
                "latent": motion.tensor_metrics(
                    latent_cache[index][native_name], latent_cache[index][bridge_name]
                ),
                "decoded": decoded_pair_metrics(
                    frame_cache[index][native_name], frame_cache[index][bridge_name]
                ),
            }
        del joint_dit
        gc.collect()
        torch.cuda.empty_cache()

    for index in range(len(prompts)):
        contact_sheet = output_dir / f"prompt_{index:02d}" / "comparison_contact_sheet.jpg"
        make_contact_sheet(contact_sheet, frame_cache[index])
        prompt_reports[index]["contact_sheet"] = str(contact_sheet.resolve())
        torch.save(latent_cache[index], output_dir / f"prompt_{index:02d}" / "rollout_latents.pt")

    system_names = list(prompt_reports[0]["systems"])
    summary = {
        "systems": {
            name: {
                "inference_seconds": float(
                    sum(row["systems"][name]["inference_seconds"] for row in prompt_reports)
                    / len(prompt_reports)
                ),
                "latent_motion": average(
                    [row["systems"][name]["latent_motion"] for row in prompt_reports]
                ),
                "decoded_motion": average(
                    [row["systems"][name]["decoded_motion"] for row in prompt_reports]
                ),
            }
            for name in system_names
        },
        "conditions": {
            label: {
                metric: average([row["conditions"][label][metric] for row in prompt_reports])
                for metric in prompt_reports[0]["conditions"][label]
            }
            for label, _ in checkpoints
        },
        "comparisons": {
            name: {
                "latent": average([row["comparisons"][name]["latent"] for row in prompt_reports]),
                "decoded": average([row["comparisons"][name]["decoded"] for row in prompt_reports]),
            }
            for name in prompt_reports[0]["comparisons"]
        },
    }
    report = {
        "protocol": {
            "resolution": [args.height, args.width],
            "num_frames": args.num_frames,
            "seed": args.seed,
            "schedule": "released monolithic 20 first-unit steps, 10 video-unit steps per stage",
            "guidance": "7.0 first unit, 5.0 video units",
            "same_noise_and_transition_noise": True,
            "external_anchor": bool(args.external_anchor),
            "anchor_source": (
                "released native-CFG unit zero reused by every system"
                if args.external_anchor
                else "none; every system generates unit zero"
            ),
        },
        "checkpoints": checkpoint_metadata,
        "prompts": prompt_reports,
        "summary": summary,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved joint monolithic evaluation: {report_path}", flush=True)


if __name__ == "__main__":
    main()
