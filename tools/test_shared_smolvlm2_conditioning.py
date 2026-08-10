#!/usr/bin/env python
"""Validate the one-forward Mobile-OV conditioner before training its image head.

The test proves that Exp1 NeoDragon conditioning is numerically unchanged. It
also validates shapes and finite values for an external-feature DreamLite head;
it deliberately does not claim visual parity for a legacy image checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import (
    MobileOVDreamLiteImageBridge,
    MobileOVNeodragonTextBridge,
    SharedMobileOVGenerationConditioner,
)
from new_mobile_ov.config import load_config


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name).lower() in {"bf16", "bfloat16"} else torch.float16


def load_state(path: str | Path) -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a bridge state dict.")
    return state, int(payload.get("step", -1))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v7.yaml")
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--image-bridge-ckpt", required=True)
    parser.add_argument("--video-bridge-ckpt", required=True)
    parser.add_argument(
        "--prompt",
        default="A red fox walking through gentle snowfall, cinematic wildlife footage.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Suffix shared by the legacy video and one-forward conditioning paths.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="output/shared_smolvlm2_conditioning_test.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_cfg = load_config(args.image_config)
    video_cfg = load_config(args.video_config)
    dtype = dtype_from_name(video_cfg.backend.dtype) if device.type == "cuda" else torch.float32
    image_state, image_step = load_state(args.image_bridge_ckpt)
    video_state, video_step = load_state(args.video_bridge_ckpt)

    video_bridge = MobileOVNeodragonTextBridge(video_cfg.bridge, device=device, dtype=dtype).eval()
    missing, unexpected = video_bridge.load_state_dict(video_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Video bridge checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    source_dim = int(video_bridge.token_bridge.smolvlm2_model.config.text_config.hidden_size)
    image_bridge = MobileOVDreamLiteImageBridge(
        image_cfg.bridge,
        image_cfg.dreamlite_bridge,
        device=device,
        dtype=dtype,
        load_feature_provider=False,
        external_feature_dim=source_dim,
    ).eval()
    image_bridge.load_trainable_state_dict(image_state)
    conditioner = SharedMobileOVGenerationConditioner(
        image_bridge=image_bridge,
        video_bridge=video_bridge,
        prompt_suffix=args.prompt_suffix,
    ).eval()
    prompts = [args.prompt]

    with torch.no_grad():
        canonical_prompts = [prompt + args.prompt_suffix for prompt in prompts]
        legacy_video = video_bridge.encode(canonical_prompts)
        shared = conditioner.encode_generation(prompts)

    video_max_abs = {
        "prompt_embeds": max_abs(legacy_video[0], shared.video_prompt_embeds),
        "prompt_mask": max_abs(legacy_video[1], shared.video_prompt_mask),
        "pooled": max_abs(legacy_video[2], shared.video_pooled),
    }
    if max(video_max_abs.values()) > 1e-6:
        raise AssertionError(f"Shared video condition lost Exp1 parity: {video_max_abs}")
    image = shared.image
    if not torch.isfinite(image.prompt_embeds).all() or not torch.isfinite(image.attention_mask).all():
        raise AssertionError("Shared DreamLite condition contains non-finite values.")

    legacy_image_bridge = MobileOVDreamLiteImageBridge(
        image_cfg.bridge,
        image_cfg.dreamlite_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    legacy_image_bridge.load_trainable_state_dict(image_state)
    for _ in range(args.warmup):
        with torch.no_grad():
            legacy_image_bridge(prompts, mode="generate")
            video_bridge.encode(canonical_prompts)
            conditioner.encode_generation(prompts)
    legacy_times = []
    shared_times = []
    for _ in range(args.runs):
        with torch.no_grad():
            legacy_times.append(
                timed(
                    device,
                    lambda: (
                        legacy_image_bridge(prompts, mode="generate"),
                        video_bridge.encode(canonical_prompts),
                    ),
                )
            )
            shared_times.append(timed(device, lambda: conditioner.encode_generation(prompts)))

    payload = {
        "device": str(device),
        "image_bridge_step": image_step,
        "video_bridge_step": video_step,
        "prompt_suffix": args.prompt_suffix,
        "shared_smolvlm2_instances": 1,
        "legacy_smolvlm2_instances": 2,
        "video_condition_max_abs_error": video_max_abs,
        "shared_image_condition_shape": list(image.prompt_embeds.shape),
        "shared_image_mask_shape": list(image.attention_mask.shape),
        "latency_seconds": {
            "legacy_two_forward_mean": statistics.mean(legacy_times),
            "shared_one_forward_mean": statistics.mean(shared_times),
            "saved_mean": statistics.mean(legacy_times) - statistics.mean(shared_times),
            "speedup_legacy_over_shared": statistics.mean(legacy_times) / statistics.mean(shared_times),
            "runs": args.runs,
        },
        "quality_status": {
            "video": "validated_exact_parity",
            "image": (
                "legacy image checkpoint is structurally compatible only; re-distill an image head "
                "on canonical shared features before visual deployment"
            ),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
