#!/usr/bin/env python
"""Measure released NeoDragon Hybrid rollout latency after skipping selected blocks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend
from new_mobile_ov.generation.neodragon_compat import install_neodragon_generation_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--skip-blocks", default="14,16")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", default="output/neodragon_block_ablation_20260811/latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires one CUDA GPU.")
    skip_blocks = tuple(int(value) for value in args.skip_blocks.split(",") if value)
    if len(set(skip_blocks)) != len(skip_blocks):
        raise ValueError("--skip-blocks must not contain duplicates.")

    device = torch.device("cuda")
    backend = build_generation_backend(load_config(args.config).backend, device=device)
    install_neodragon_generation_patches(device=None)
    dit = backend.pipeline.dit
    original_blocks = list(dit.transformer_blocks)
    if min(skip_blocks) < 0 or max(skip_blocks) >= len(original_blocks):
        raise ValueError(f"--skip-blocks must be within [0, {len(original_blocks) - 1}].")

    prompt = "A red fox walking through gentle snowfall, cinematic wildlife footage, cinematic, realistic textures, high detail, natural colours"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prompt_embeds, prompt_mask, pooled = backend.encode_neodragon_context([prompt])
        first_frame = backend.pipeline.first_frame_gen_pipeline(
            prompt=prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        ).images[0].convert("RGB")

    def rollout(skipped: tuple[int, ...]) -> float:
        dit.transformer_blocks = torch.nn.ModuleList(
            [block for index, block in enumerate(original_blocks) if index not in skipped]
        )
        torch.manual_seed(args.seed + 100)
        torch.cuda.manual_seed_all(args.seed + 100)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            backend.generate_video_from_bridge_condition(
                prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                first_frame=first_frame,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            )
        torch.cuda.synchronize()
        return time.perf_counter() - start

    try:
        # Warm-up outside the reported measurements.
        rollout(())
        rollout(skip_blocks)
        baseline = [rollout(()) for _ in range(args.repeats)]
        pruned = [rollout(skip_blocks) for _ in range(args.repeats)]
    finally:
        dit.transformer_blocks = torch.nn.ModuleList(original_blocks)

    baseline_mean = sum(baseline) / len(baseline)
    pruned_mean = sum(pruned) / len(pruned)
    result = {
        "protocol": "Same native condition, first frame, seed, resolution, and 1-1-1 rollout; VAE decode is included.",
        "original_blocks": len(original_blocks),
        "skipped_blocks": list(skip_blocks),
        "baseline_seconds": baseline,
        "pruned_seconds": pruned,
        "baseline_mean_seconds": baseline_mean,
        "pruned_mean_seconds": pruned_mean,
        "speedup": baseline_mean / pruned_mean,
        "latency_reduction_percent": 100.0 * (1.0 - pruned_mean / baseline_mean),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
