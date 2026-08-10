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


class CudaModuleTimer:
    """Collect CUDA-event timings without synchronizing each model call."""

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def measure(self, name: str, fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.records.setdefault(name, []).append((start, end))
        return result

    def seconds_by_name(self) -> dict[str, float]:
        synchronize(torch.device("cuda"))
        return {
            name: sum(start.elapsed_time(end) for start, end in events) / 1000.0
            for name, events in self.records.items()
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
    parser.add_argument(
        "--block-noise",
        choices=("released", "cpu", "cuda"),
        default="released",
        help="Use released sampler or the validated Mobile-OV CPU/CUDA sampler.",
    )
    parser.add_argument(
        "--breakdown-runs",
        type=int,
        default=0,
        help="Additional Mobile-condition rollouts for a DiT/VAE CUDA-event breakdown.",
    )
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
    if args.block_noise != "released":
        from new_mobile_ov.generation.neodragon_compat import (
            install_neodragon_generation_patches,
        )

        install_neodragon_generation_patches(
            device=device if args.block_noise == "cuda" else None
        )

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
    breakdown_runs: list[dict[str, float]] = []
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

        if args.breakdown_runs:
            import neodragon.utils.generation_utils as generation_utils

            original_dit_forward = backend.pipeline.dit.forward
            original_vae_encode = backend.pipeline.vae.encode
            original_vae_decode = backend.pipeline.vae.decode
            original_block_noise = generation_utils._sample_block_noise
            try:
                for index in range(args.breakdown_runs):
                    module_timer = CudaModuleTimer()
                    block_noise_seconds: list[float] = []

                    def timed_dit(*call_args, **call_kwargs):
                        return module_timer.measure(
                            "dit", lambda: original_dit_forward(*call_args, **call_kwargs)
                        )

                    def timed_vae_encode(*call_args, **call_kwargs):
                        return module_timer.measure(
                            "vae_encode", lambda: original_vae_encode(*call_args, **call_kwargs)
                        )

                    def timed_vae_decode(*call_args, **call_kwargs):
                        return module_timer.measure(
                            "vae_decode", lambda: original_vae_decode(*call_args, **call_kwargs)
                        )

                    def timed_block_noise(*call_args, **call_kwargs):
                        started = time.perf_counter()
                        result = original_block_noise(*call_args, **call_kwargs)
                        block_noise_seconds.append(time.perf_counter() - started)
                        return result

                    backend.pipeline.dit.forward = timed_dit
                    backend.pipeline.vae.encode = timed_vae_encode
                    backend.pipeline.vae.decode = timed_vae_decode
                    generation_utils._sample_block_noise = timed_block_noise
                    total_seconds = timed(
                        device,
                        lambda index=index: rollout(mobile_condition, args.seed + 3000 + index),
                    )
                    components = module_timer.seconds_by_name()
                    measured_seconds = sum(components.values())
                    components["other"] = total_seconds - measured_seconds
                    components["total"] = total_seconds
                    components["dit_calls"] = float(len(module_timer.records.get("dit", [])))
                    components["block_noise_cpu"] = sum(block_noise_seconds)
                    components["block_noise_calls"] = float(len(block_noise_seconds))
                    breakdown_runs.append(components)
                    print(json.dumps({"breakdown": components}), flush=True)
            finally:
                backend.pipeline.dit.forward = original_dit_forward
                backend.pipeline.vae.encode = original_vae_encode
                backend.pipeline.vae.decode = original_vae_decode
                generation_utils._sample_block_noise = original_block_noise

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
            "block_noise": args.block_noise,
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
    if breakdown_runs:
        payload["mobile_rollout_breakdown_seconds"] = {
            name: summary([row.get(name, 0.0) for row in breakdown_runs])
            for name in sorted({key for row in breakdown_runs for key in row})
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
