#!/usr/bin/env python
"""Isolate DreamLite image conditioning from NeoDragon text/video conditioning.

For every prompt and fixed seed, the experiment renders two DreamLite anchors
(``native_qwen`` and ``v8_imageonly``) and combines each with both NeoDragon
text conditions (native ContextAdapter and Exp1-64K bridge).  The four video
cells form a small causal ablation rather than an end-to-end comparison alone.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from decord import VideoReader
from diffusers.utils import export_to_video
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import (  # noqa: E402
    MobileOVDreamLiteImageBridge,
    MobileOVNeodragonTextBridge,
)
from new_mobile_ov.bridge.dreamlite_image_bridge import DreamLiteCondition  # noqa: E402
from new_mobile_ov.checkpoints import ensure_neodragon_assets  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.generation import build_generation_backend  # noqa: E402
from new_mobile_ov.generation.backends import DreamLiteMobileBackend  # noqa: E402
from new_mobile_ov.generation.neodragon_compat import (  # noqa: E402
    install_neodragon_generation_patches,
)
from new_mobile_ov.training.dreamlite_distillation import (  # noqa: E402
    DreamLiteFrozenQwenTeacher,
    dreamlite_content_aware_representation_losses,
    dreamlite_direct_representation_losses,
)


@dataclass(frozen=True)
class PromptItem:
    name: str
    prompt: str


def safe_stem(value: str, max_length: int = 72) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip().replace(" ", "_")
    return (normalized[:max_length] or "prompt").strip("_")


def load_prompts(path: Path, max_prompts: int) -> list[PromptItem]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"name", "prompt"}.issubset(reader.fieldnames):
            raise ValueError(f"Prompt CSV must contain name,prompt: {path}")
        prompts = [
            PromptItem(
                name=safe_stem(row["name"]),
                prompt=" ".join(row["prompt"].strip().split()),
            )
            for row in reader
            if row.get("name") and row.get("prompt") and row["prompt"].strip()
        ]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No valid prompts read from {path}")
    return prompts


def cpu_condition(condition: DreamLiteCondition) -> DreamLiteCondition:
    return DreamLiteCondition(
        condition.prompt_embeds.detach().float().cpu(),
        condition.attention_mask.detach().cpu(),
    )


def device_condition(
    condition: DreamLiteCondition,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DreamLiteCondition:
    return DreamLiteCondition(
        condition.prompt_embeds.to(device=device, dtype=dtype),
        condition.attention_mask.to(device=device),
    )


def release() -> None:
    """Return cached CUDA blocks after callers explicitly release model references."""

    gc.collect()
    torch.cuda.empty_cache()


def image_path(root: Path, image_condition: str, index: int, item: PromptItem) -> Path:
    return root / "anchors" / image_condition / f"{index:02d}_{item.name}.png"


def video_path(
    root: Path,
    image_condition: str,
    video_condition: str,
    index: int,
    item: PromptItem,
) -> Path:
    return root / "videos" / f"image_{image_condition}__text_{video_condition}" / f"{index:02d}_{item.name}.mp4"


def valid_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with Image.open(path) as value:
            return value.size == expected_size
    except Exception:
        return False


def valid_video(path: Path, expected_frames: int) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        return len(VideoReader(str(path), num_threads=1)) == expected_frames
    except Exception:
        return False


def load_checkpoint_state(path: Path, key: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get(key, payload)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not expose a state dictionary under {key!r}.")
    metadata = {
        "path": str(path.resolve()),
        "step": int(payload.get("step", -1)),
        "target": payload.get("target"),
        "architecture": payload.get("architecture"),
    }
    return state, metadata


def scalar_metrics(student: DreamLiteCondition, teacher: DreamLiteCondition) -> dict[str, float]:
    direct = dreamlite_direct_representation_losses(student, teacher)
    content = dreamlite_content_aware_representation_losses(student, teacher)
    keys = {
        "token_normalized_mse": direct["token_normalized_mse"],
        "token_cosine_distance": direct["token_cosine"],
        "pooled_cosine_distance": direct["pooled_cosine"],
        "mask_agreement": direct["mask_agreement"],
        "content_token_normalized_mse": content["content_token_normalized_mse"],
        "content_token_cosine_distance": content["content_token_cosine"],
        "content_pooled_cosine_distance": content["content_pooled_cosine"],
    }
    return {
        **{name: float(value.item()) for name, value in keys.items()},
        "student_tokens": int(student.attention_mask.sum().item()),
        "teacher_tokens": int(teacher.attention_mask.sum().item()),
    }


@torch.inference_mode()
def make_image_conditions(
    image_cfg,
    prompts: list[PromptItem],
    *,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, list[DreamLiteCondition]], list[dict[str, float]], dict[str, object]]:
    prompt_texts = [item.prompt for item in prompts]
    print("Encoding frozen native Qwen3-VL DreamLite conditions.", flush=True)
    teacher = DreamLiteFrozenQwenTeacher(image_cfg.dreamlite, device=device, dtype=dtype)
    # Match deployed VBench inference exactly: each prompt is conditioned as a
    # single-item batch, so variable-length masks cannot be affected by a longer
    # neighbouring prompt in this diagnostic batch.
    native_conditions = [
        cpu_condition(teacher.encode([prompt], mode="generate"))
        for prompt in prompt_texts
    ]
    del teacher
    release()

    state, checkpoint_metadata = load_checkpoint_state(checkpoint, "bridge")
    print(f"Encoding V8 image-only bridge condition from {checkpoint}.", flush=True)
    bridge = MobileOVDreamLiteImageBridge(
        image_cfg.bridge,
        image_cfg.dreamlite_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    bridge.load_trainable_state_dict(state)
    student_conditions = [
        cpu_condition(bridge([prompt], mode="generate"))
        for prompt in prompt_texts
    ]
    del bridge, state
    release()

    conditions = {
        "native_qwen": native_conditions,
        "v8_imageonly": student_conditions,
    }
    metrics = [
        scalar_metrics(conditions["v8_imageonly"][index], conditions["native_qwen"][index])
        for index in range(len(prompts))
    ]
    return conditions, metrics, checkpoint_metadata


@torch.inference_mode()
def generate_anchors(
    image_cfg,
    conditions: dict[str, list[DreamLiteCondition]],
    prompts: list[PromptItem],
    *,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    time_id_width: int,
    time_id_height: int,
    image_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, list[Image.Image]], dict[str, list[float]]]:
    backend = DreamLiteMobileBackend(image_cfg.dreamlite, device=device, dtype=dtype, load_vae=True)
    anchors: dict[str, list[Image.Image]] = {name: [] for name in conditions}
    timings: dict[str, list[float]] = {name: [] for name in conditions}
    expected_size = (width, height)
    for index, item in enumerate(prompts):
        for condition_name, values in conditions.items():
            destination = image_path(output_dir, condition_name, index + 1, item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if valid_image(destination, expected_size):
                image = Image.open(destination).convert("RGB")
                elapsed = 0.0
            else:
                started = time.perf_counter()
                image = backend.generate_images(
                    device_condition(values[index], device=device, dtype=dtype),
                    width=width,
                    height=height,
                    time_id_width=time_id_width,
                    time_id_height=time_id_height,
                    num_steps=image_steps,
                    seed=seed + index,
                )[0].convert("RGB")
                elapsed = time.perf_counter() - started
                temporary = destination.with_suffix(".tmp.png")
                image.save(temporary)
                os.replace(temporary, destination)
            anchors[condition_name].append(image)
            timings[condition_name].append(elapsed)
        print(
            f"Anchors {index + 1}/{len(prompts)} prompt={item.name} "
            f"native={timings['native_qwen'][-1]:.2f}s student={timings['v8_imageonly'][-1]:.2f}s",
            flush=True,
        )
    del backend
    release()
    return anchors, timings


@torch.inference_mode()
def make_video_conditions(
    video_cfg,
    prompts: list[PromptItem],
    *,
    checkpoint: Path,
    backend,
    modifier: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], dict[str, object]]:
    prompt_texts = [item.prompt + modifier for item in prompts]
    # The regular VBench generator encodes one prompt per call. Retaining that
    # behaviour makes this control robust to any tokenizer padding differences.
    native_conditions = [
        tuple(value.detach().cpu() for value in backend.encode_neodragon_context([prompt]))
        for prompt in prompt_texts
    ]

    state, metadata = load_checkpoint_state(checkpoint, "bridge")
    bridge = MobileOVNeodragonTextBridge(video_cfg.bridge, device=device, dtype=dtype).eval()
    missing, unexpected = bridge.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Exp1 video checkpoint does not match MobileOVNeodragonTextBridge: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
    )
    student_conditions = [
        tuple(value.detach().cpu() for value in bridge.encode([prompt]))
        for prompt in prompt_texts
    ]
    del bridge, state
    release()
    return {"native_neodragon": native_conditions, "exp1_64k": student_conditions}, metadata


def to_video_device(
    condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, mask, pooled = condition
    return tokens.to(device=device, dtype=dtype), mask.to(device=device), pooled.to(device=device, dtype=dtype)


@torch.inference_mode()
def generate_videos(
    backend,
    anchors: dict[str, list[Image.Image]],
    video_conditions: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    prompts: list[PromptItem],
    *,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {}
    for image_name, image_values in anchors.items():
        for text_name, text_values in video_conditions.items():
            branch = f"image_{image_name}__text_{text_name}"
            timings[branch] = []
            for index, item in enumerate(prompts):
                destination = video_path(output_dir, image_name, text_name, index + 1, item)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if valid_video(destination, num_frames):
                    elapsed = 0.0
                else:
                    prompt_embeds, prompt_mask, pooled = to_video_device(
                        text_values[index], device=device, dtype=dtype
                    )
                    torch.manual_seed(seed + index)
                    torch.cuda.manual_seed_all(seed + index)
                    started = time.perf_counter()
                    frames = backend.generate_video_from_bridge_condition(
                        item.prompt,
                        prompt_embeds=prompt_embeds,
                        prompt_mask=prompt_mask,
                        pooled_prompt_embeds=pooled,
                        first_frame=image_values[index],
                        width=width,
                        height=height,
                        num_frames=num_frames,
                    )
                    elapsed = time.perf_counter() - started
                    temporary = destination.with_suffix(".tmp.mp4")
                    export_to_video(frames, temporary, fps=fps)
                    os.replace(temporary, destination)
                timings[branch].append(elapsed)
                print(
                    f"Videos {branch} {index + 1}/{len(prompts)} prompt={item.name} elapsed={elapsed:.2f}s",
                    flush=True,
                )
    return timings


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default="configs/prompts/dreamlite_v8_native_video_control_20.csv")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v8.yaml")
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--image-bridge-checkpoint", required=True, type=Path)
    parser.add_argument("--video-bridge-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--anchor-width", type=int, default=1024)
    parser.add_argument("--anchor-height", type=int, default=640)
    parser.add_argument("--anchor-time-id-width", type=int, default=1280)
    parser.add_argument("--anchor-time-id-height", type=int, default=800)
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This controlled ablation requires one CUDA GPU.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prompts = load_prompts(args.prompts, args.max_prompts)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    image_cfg = load_config(args.image_config)
    video_cfg = load_config(args.video_config)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_conditions, condition_metrics, image_checkpoint = make_image_conditions(
        image_cfg,
        prompts,
        checkpoint=args.image_bridge_checkpoint,
        device=device,
        dtype=dtype,
    )
    anchors, anchor_seconds = generate_anchors(
        image_cfg,
        image_conditions,
        prompts,
        output_dir=output_dir,
        seed=args.seed,
        width=args.anchor_width,
        height=args.anchor_height,
        time_id_width=args.anchor_time_id_width,
        time_id_height=args.anchor_time_id_height,
        image_steps=args.image_steps,
        device=device,
        dtype=dtype,
    )
    del image_conditions
    release()

    repo_path, _, _ = ensure_neodragon_assets(
        repo_path=video_cfg.backend.extra.get("repo_path"),
        cache_dir=video_cfg.backend.extra.get("cache_dir"),
        model_id=video_cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=video_cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    install_neodragon_generation_patches()
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    backend = build_generation_backend(video_cfg.backend, device=device)
    video_conditions, video_checkpoint = make_video_conditions(
        video_cfg,
        prompts,
        checkpoint=args.video_bridge_checkpoint,
        backend=backend,
        modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
        dtype=dtype,
    )
    video_seconds = generate_videos(
        backend,
        anchors,
        video_conditions,
        prompts,
        output_dir=output_dir,
        seed=args.seed,
        width=args.video_width,
        height=args.video_height,
        num_frames=args.num_frames,
        fps=args.fps,
        device=device,
        dtype=dtype,
    )
    del backend, video_conditions, anchors
    release()

    summary = {
        "status": "ok",
        "protocol": {
            "image_conditions": ["native_qwen", "v8_imageonly"],
            "video_conditions": ["native_neodragon", "exp1_64k"],
            "cells": [
                "image_native_qwen__text_native_neodragon",
                "image_native_qwen__text_exp1_64k",
                "image_v8_imageonly__text_native_neodragon",
                "image_v8_imageonly__text_exp1_64k",
            ],
            "controls": (
                "Every cell for a prompt uses the same DreamLite seed, NeoDragon seed, render size, "
                "logical time_ids, released DreamLite generator, released Hybrid NeoDragon DiT, and "
                "fixed Exp1-64K checkpoint when the Exp1 text condition is selected."
            ),
            "interpretation": {
                "native_qwen_vs_v8_imageonly": "DreamLite image-bridge alignment and anchor generation effect.",
                "native_neodragon_vs_exp1_64k": "NeoDragon video text-condition effect with an identical anchor.",
                "full_pipeline": "V8 image bridge plus Exp1-64K video bridge under fixed released generators.",
            },
        },
        "image_checkpoint": image_checkpoint,
        "video_checkpoint": video_checkpoint,
        "seed": args.seed,
        "image_render": {
            "width": args.anchor_width,
            "height": args.anchor_height,
            "logical_width": args.anchor_time_id_width,
            "logical_height": args.anchor_time_id_height,
            "steps": args.image_steps,
        },
        "video_render": {
            "width": args.video_width,
            "height": args.video_height,
            "frames": args.num_frames,
            "fps": args.fps,
        },
        "mean_condition_metrics": {
            key: mean([row[key] for row in condition_metrics])
            for key in condition_metrics[0]
            if isinstance(condition_metrics[0][key], float)
        },
        "mean_anchor_seconds": {name: mean(values) for name, values in anchor_seconds.items()},
        "mean_video_seconds": {name: mean(values) for name, values in video_seconds.items()},
        "prompts": [
            {
                "index": index + 1,
                "name": item.name,
                "prompt": item.prompt,
                "seed": args.seed + index,
                "condition_metrics_v8_vs_native": condition_metrics[index],
                "anchors": {
                    name: str(image_path(output_dir, name, index + 1, item))
                    for name in ("native_qwen", "v8_imageonly")
                },
                "videos": {
                    f"image_{image_name}__text_{text_name}": str(
                        video_path(output_dir, image_name, text_name, index + 1, item)
                    )
                    for image_name in ("native_qwen", "v8_imageonly")
                    for text_name in ("native_neodragon", "exp1_64k")
                },
            }
            for index, item in enumerate(prompts)
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
