#!/usr/bin/env python
"""Compare native and Mobile-OV NeoDragon conditions under the same rollout.

This intentionally removes first-frame and text-encoder timing from the DiT
comparison. Both conditions are precomputed, then injected into the same
released Hybrid DiT/VAE with the same native first frame and seed.
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

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed(device: torch.device, fn) -> float:
    synchronize(device)
    started = time.perf_counter()
    frames = fn()
    synchronize(device)
    del frames
    return time.perf_counter() - started


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


def load_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a bridge state.")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--video-bridge-ckpt",
        default=(
            "checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/"
            "neodragon_text_bridge_latest.pt"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="A red fox walking through gentle snowfall, cinematic wildlife footage.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=", cinematic, realistic textures, high detail, natural colours",
    )
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--output",
        default="output/neodragon_controlled_condition_latency/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Controlled NeoDragon benchmark requires one CUDA GPU.")
    if args.warmup_pairs < 0 or args.pairs < 2:
        raise ValueError("--warmup-pairs must be non-negative and --pairs must be at least two.")

    cfg = load_config(args.config)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    backend = build_generation_backend(cfg.backend, device=device)

    canonical_prompt = str(args.prompt) + str(args.prompt_suffix)
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    missing, unexpected = bridge.load_state_dict(
        load_state(args.video_bridge_ckpt),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Video bridge checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        native_prompt_embeds, native_mask, native_pooled = backend.encode_neodragon_context(
            [canonical_prompt]
        )
        mobile_prompt_embeds, mobile_mask, mobile_pooled = bridge.encode(
            [canonical_prompt]
        )
        first_frame = backend.pipeline.first_frame_gen_pipeline(
            prompt=canonical_prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        ).images[0].convert("RGB")

    shape_report = {
        "native_prompt_embeds": list(native_prompt_embeds.shape),
        "mobile_prompt_embeds": list(mobile_prompt_embeds.shape),
        "native_mask": list(native_mask.shape),
        "mobile_mask": list(mobile_mask.shape),
        "native_pooled": list(native_pooled.shape),
        "mobile_pooled": list(mobile_pooled.shape),
        "same_dit_condition_shapes": (
            tuple(native_prompt_embeds.shape) == tuple(mobile_prompt_embeds.shape)
            and tuple(native_mask.shape) == tuple(mobile_mask.shape)
            and tuple(native_pooled.shape) == tuple(mobile_pooled.shape)
        ),
    }
    if not shape_report["same_dit_condition_shapes"]:
        raise RuntimeError(f"Condition shapes differ: {shape_report}")

    def rollout(condition, seed: int):
        prompt_embeds, prompt_mask, pooled = condition
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return backend.generate_video_from_bridge_condition(
            args.prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            first_frame=first_frame,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

    native_condition = (native_prompt_embeds, native_mask, native_pooled)
    mobile_condition = (mobile_prompt_embeds, mobile_mask, mobile_pooled)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for index in range(args.warmup_pairs):
            timed(device, lambda index=index: rollout(native_condition, args.seed + index))
            timed(device, lambda index=index: rollout(mobile_condition, args.seed + index))

        pairs: list[dict[str, float]] = []
        for index in range(args.pairs):
            seed = args.seed + 1000 + index
            # Alternate order so cache/order effects do not favour one condition.
            if index % 2 == 0:
                native_seconds = timed(device, lambda: rollout(native_condition, seed))
                mobile_seconds = timed(device, lambda: rollout(mobile_condition, seed))
            else:
                mobile_seconds = timed(device, lambda: rollout(mobile_condition, seed))
                native_seconds = timed(device, lambda: rollout(native_condition, seed))
            pairs.append(
                {
                    "pair": index,
                    "native_seconds": native_seconds,
                    "mobile_seconds": mobile_seconds,
                    "mobile_minus_native_seconds": mobile_seconds - native_seconds,
                }
            )
            print(json.dumps(pairs[-1]), flush=True)

    native_seconds = [row["native_seconds"] for row in pairs]
    mobile_seconds = [row["mobile_seconds"] for row in pairs]
    deltas = [row["mobile_minus_native_seconds"] for row in pairs]
    payload = {
        "protocol": {
            "first_frame": "one shared native SSD1B first frame",
            "seed": "identical within each native/mobile pair",
            "video_backend": "same released NeoDragon Hybrid DiT, VAE, scheduler, resolution, and frame count",
            "conditions": "precomputed native ContextAdapter output versus precomputed Exp1 Mobile-OV MCP output",
            "order": "alternated per pair",
            "mp4_encoding": "excluded",
        },
        "prompt": args.prompt,
        "prompt_suffix": args.prompt_suffix,
        "shape_report": shape_report,
        "native_rollout_seconds": summary(native_seconds),
        "mobile_rollout_seconds": summary(mobile_seconds),
        "paired_delta_mobile_minus_native_seconds": summary(deltas),
        "pairs": pairs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
