#!/usr/bin/env python
"""Select a low-impact NeoDragon block pair using controlled rollout ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend
from new_mobile_ov.generation.neodragon_compat import install_neodragon_generation_patches


DEFAULT_PROMPTS = [
    "A red fox walking through gentle snowfall, cinematic wildlife footage.",
    "A golden retriever runs along a sunny beach and splashes through shallow waves.",
    "A chef in a white apron tosses vegetables in a hot wok inside a busy restaurant kitchen.",
    "A small astronaut carefully walks across a dusty red Martian landscape while a spacecraft rises behind them.",
    "A vintage red convertible drives through a rainy city street at night, reflected neon lights on the road.",
    "A hummingbird hovers beside bright purple flowers in a lush tropical garden, natural documentary style.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--calibration-prompts", type=int, default=4)
    parser.add_argument("--pair-candidates", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        default="output/neodragon_block_ablation",
    )
    return parser.parse_args()


def frames_to_array(frames) -> np.ndarray:
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0 for frame in frames])


def drift_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = candidate - reference
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def save_comparison_sheet(
    rows: list[dict[str, object]],
    destination: Path,
) -> None:
    frame_indices = (0, 24, 48)
    thumb_width, thumb_height = 192, 120
    header = 36
    row_height = header + thumb_height
    canvas = Image.new("RGB", (thumb_width * 6, row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(rows):
        top = row_index * row_height
        prompt = str(row["prompt"])
        draw.text((6, top + 6), prompt[:115], fill="black")
        for group_index, frame_group in enumerate((row["baseline"], row["pruned"])):
            for frame_offset, frame_index in enumerate(frame_indices):
                image = frame_group[frame_index].convert("RGB").resize(
                    (thumb_width, thumb_height), Image.Resampling.LANCZOS
                )
                x = (group_index * len(frame_indices) + frame_offset) * thumb_width
                canvas.paste(image, (x, top + header))
        draw.text((6, top + header + 2), "baseline", fill="white")
        draw.text((thumb_width * 3 + 6, top + header + 2), "16-block", fill="white")

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This ablation requires one CUDA GPU.")
    if not 2 <= args.calibration_prompts <= len(DEFAULT_PROMPTS):
        raise ValueError("--calibration-prompts must select between 2 and 6 prompts.")
    if args.pair_candidates < 2:
        raise ValueError("--pair-candidates must be at least two.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    backend = build_generation_backend(config.backend, device=device)
    # Matches the intended covariance while avoiding the released Python loop.
    install_neodragon_generation_patches(device=None)
    dit = backend.pipeline.dit
    original_blocks = list(dit.transformer_blocks)
    prompts = DEFAULT_PROMPTS
    calibration = prompts[: args.calibration_prompts]

    conditions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for index, prompt in enumerate(prompts):
            native_prompt = prompt + ", cinematic, realistic textures, high detail, natural colours"
            condition = backend.encode_neodragon_context([native_prompt])
            first_frame = backend.pipeline.first_frame_gen_pipeline(
                prompt=native_prompt,
                num_images_per_prompt=1,
                generator=torch.Generator(device=device).manual_seed(args.seed + index),
            ).images[0].convert("RGB")
            conditions[prompt] = (*condition, first_frame)

    def rollout(prompt: str, *, skip_blocks: tuple[int, ...], seed: int):
        selected = [block for index, block in enumerate(original_blocks) if index not in skip_blocks]
        dit.transformer_blocks = torch.nn.ModuleList(selected)
        prompt_embeds, prompt_mask, pooled, first_frame = conditions[prompt]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return backend.generate_video_from_bridge_condition(
            prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            first_frame=first_frame,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

    try:
        baselines: dict[str, tuple[object, np.ndarray]] = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for prompt_index, prompt in enumerate(prompts):
                frames = rollout(prompt, skip_blocks=(), seed=args.seed + 100 + prompt_index)
                baselines[prompt] = (frames, frames_to_array(frames))

            per_block: dict[int, list[dict[str, float]]] = {index: [] for index in range(len(original_blocks))}
            for block_index in range(len(original_blocks)):
                for prompt_index, prompt in enumerate(calibration):
                    frames = rollout(
                        prompt,
                        skip_blocks=(block_index,),
                        seed=args.seed + 100 + prompt_index,
                    )
                    per_block[block_index].append(
                        drift_metrics(baselines[prompt][1], frames_to_array(frames))
                    )
                print(
                    json.dumps(
                        {
                            "single_block": block_index,
                            "mean_mse": float(np.mean([row["mse"] for row in per_block[block_index]])),
                        }
                    ),
                    flush=True,
                )

            block_scores = [
                {
                    "block": block_index,
                    "mean_mse": float(np.mean([row["mse"] for row in scores])),
                    "mean_mae": float(np.mean([row["mae"] for row in scores])),
                    "parameters": sum(parameter.numel() for parameter in original_blocks[block_index].parameters()),
                }
                for block_index, scores in per_block.items()
            ]
            block_scores.sort(key=lambda row: row["mean_mse"])
            candidate_blocks = [row["block"] for row in block_scores[: args.pair_candidates]]

            pair_scores: list[dict[str, object]] = []
            for first_offset, first_block in enumerate(candidate_blocks):
                for second_block in candidate_blocks[first_offset + 1 :]:
                    metrics = []
                    for prompt_index, prompt in enumerate(calibration):
                        frames = rollout(
                            prompt,
                            skip_blocks=(first_block, second_block),
                            seed=args.seed + 100 + prompt_index,
                        )
                        metrics.append(drift_metrics(baselines[prompt][1], frames_to_array(frames)))
                    pair_scores.append(
                        {
                            "blocks": [first_block, second_block],
                            "mean_mse": float(np.mean([row["mse"] for row in metrics])),
                            "mean_mae": float(np.mean([row["mae"] for row in metrics])),
                        }
                    )
                    print(json.dumps({"pair": pair_scores[-1]}), flush=True)

        pair_scores.sort(key=lambda row: row["mean_mse"])
        selected_pair = tuple(int(index) for index in pair_scores[0]["blocks"])
        comparison_rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for prompt_index, prompt in enumerate(prompts):
                frames = rollout(
                    prompt,
                    skip_blocks=selected_pair,
                    seed=args.seed + 100 + prompt_index,
                )
                comparison_rows.append(
                    {
                        "prompt": prompt,
                        "baseline": baselines[prompt][0],
                        "pruned": frames,
                        "metrics": drift_metrics(baselines[prompt][1], frames_to_array(frames)),
                    }
                )
        save_comparison_sheet(comparison_rows, output_dir / "baseline_vs_16block.png")

        report = {
            "protocol": {
                "teacher": "released NeoDragon Hybrid DiT with native text condition",
                "schedule": "released Hybrid 1-1-1",
                "first_frame": "fixed native SSD1B frame per prompt",
                "seed": "fixed per prompt and shared by baseline/ablation",
                "importance": "full-rollout RGB drift after skipping a block",
                "quality_note": "drift ranks candidate blocks; visual/VBench evaluation is still required for quality claims",
            },
            "original_blocks": len(original_blocks),
            "original_block_parameters": [
                sum(parameter.numel() for parameter in block.parameters()) for block in original_blocks
            ],
            "single_block_scores": block_scores,
            "pair_candidates": candidate_blocks,
            "pair_scores": pair_scores,
            "selected_pair": list(selected_pair),
            "selected_16block_parameters": sum(
                sum(parameter.numel() for parameter in block.parameters())
                for index, block in enumerate(original_blocks)
                if index not in selected_pair
            ),
            "six_prompt_comparison": [
                {"prompt": row["prompt"], **row["metrics"]} for row in comparison_rows
            ],
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        dit.transformer_blocks = torch.nn.ModuleList(original_blocks)


if __name__ == "__main__":
    main()
