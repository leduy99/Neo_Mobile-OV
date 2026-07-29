#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import (  # noqa: E402
    MobileOVNeodragonTextBridge,
    MobileOVSSD1BImageBridge,
)
from new_mobile_ov.checkpoints import ensure_neodragon_assets  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.generation import build_generation_backend  # noqa: E402


DEFAULT_PROMPTS = (
    "A red panda eating bamboo.",
    "A surfer riding a large ocean wave.",
    "A golden retriever runs through a field of yellow flowers.",
    "A young adult dances gracefully on a sunlit beach.",
    "An astronaut explores a crystalline cave illuminated by blue light, cinematic photography.",
    "A chef prepares fresh pasta in a warm rustic kitchen while afternoon sunlight enters through the window.",
)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def safe_stem(text: str, max_len: int = 72) -> str:
    value = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (value[:max_len] or "prompt").strip("_")


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [value.strip() for value in args.prompt if value.strip()]
    if args.prompt_file:
        values = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        return [value.strip() for value in values if value.strip()]
    return list(DEFAULT_PROMPTS)


def checkpoint_state(path: Path, key: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if key not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no `{key}` state.")
    state = checkpoint[key]
    metadata = {
        "path": str(path),
        "step": int(checkpoint.get("step", -1)),
        "target": checkpoint.get("target"),
        "architecture": checkpoint.get("architecture"),
        "state_tensors": len(state),
        "state_numel": sum(value.numel() for value in state.values() if torch.is_tensor(value)),
    }
    return state, metadata


@torch.inference_mode()
def encode_image_conditions(
    prompts: list[str],
    *,
    cfg,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    append_modifier: bool,
    modifier: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, object]]:
    state, metadata = checkpoint_state(checkpoint_path, "image_bridge")
    bridge = MobileOVSSD1BImageBridge(
        cfg.bridge,
        cfg.image_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    bridge.load_trainable_state_dict(state)
    del state
    values = []
    for prompt in prompts:
        conditioned_prompt = prompt + modifier if append_modifier else prompt
        condition = bridge([conditioned_prompt])
        values.append(tuple(tensor.detach().cpu() for tensor in condition))
    del bridge
    torch.cuda.empty_cache()
    return values, metadata


@torch.inference_mode()
def generate_first_frames(
    prompts: list[str],
    conditions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    model_path: Path,
    output_dir: Path,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[Image.Image], list[float]]:
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline

    pipeline = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    images: list[Image.Image] = []
    timings: list[float] = []
    for index, (prompt, values) in enumerate(zip(prompts, conditions)):
        clip_l, clip_big_g, pooled = (
            value.to(device=device, dtype=dtype) for value in values
        )
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            image = pipeline(
                prompt_embeds=torch.cat([clip_l, clip_big_g], dim=-1),
                pooled_prompt_embeds=pooled,
                generator=torch.Generator(device=device).manual_seed(seed + index),
            ).images[0]
        timings.append(time.perf_counter() - started)
        image = image.convert("RGB")
        image.save(output_dir / f"{index + 1:02d}_{safe_stem(prompt)}_image_bridge.png")
        images.append(image)
    del pipeline
    torch.cuda.empty_cache()
    return images, timings


@torch.inference_mode()
def encode_video_conditions(
    prompts: list[str],
    *,
    cfg,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    modifier: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, object]]:
    state, metadata = checkpoint_state(checkpoint_path, "bridge")
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    missing, unexpected = bridge.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Video bridge checkpoint does not match the inference architecture: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    del state
    encoded = bridge.encode([prompt + modifier for prompt in prompts])
    values = [
        tuple(tensor[index : index + 1].detach().cpu() for tensor in encoded)
        for index in range(len(prompts))
    ]
    del bridge, encoded
    torch.cuda.empty_cache()
    return values, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Exp1 with native versus Mobile-OV Image Bridge first frames."
    )
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--image-config",
        default="configs/mobile_ov_ssd1b_image_bridge.yaml",
    )
    parser.add_argument("--video-bridge-ckpt", required=True)
    parser.add_argument("--image-bridge-ckpt", required=True)
    parser.add_argument("--output-dir", default="output/image_bridge_to_exp1_rollout")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompt-file")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--image-append-prompt-modifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="The 100k Image Bridge measured better without Neodragon's generic modifier.",
    )
    parser.add_argument(
        "--skip-native-first-frame",
        action="store_true",
        help="Only run the Image Bridge first-frame path.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Combined image/video generation requires a CUDA allocation.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    prompts = load_prompts(args)
    if not prompts:
        raise ValueError("At least one prompt is required.")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "first_frames"
    native_video_dir = output_dir / "exp1_native_first_frame"
    combined_video_dir = output_dir / "image_bridge_first_frame"
    for path in (image_dir, native_video_dir, combined_video_dir):
        path.mkdir(parents=True, exist_ok=True)

    video_cfg = load_config(args.video_config)
    image_cfg = load_config(args.image_config)
    repo_path, _, model_path = ensure_neodragon_assets(
        repo_path=video_cfg.backend.extra.get("repo_path"),
        cache_dir=video_cfg.backend.extra.get("cache_dir"),
        model_id=video_cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=video_cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    image_conditions, image_metadata = encode_image_conditions(
        prompts,
        cfg=image_cfg,
        checkpoint_path=Path(args.image_bridge_ckpt),
        device=device,
        dtype=dtype,
        append_modifier=args.image_append_prompt_modifier,
        modifier=DEFAULT_PROMPT_MODIFIER,
    )
    first_frames, first_frame_seconds = generate_first_frames(
        prompts,
        image_conditions,
        model_path=Path(model_path),
        output_dir=image_dir,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    del image_conditions

    video_conditions, video_metadata = encode_video_conditions(
        prompts,
        cfg=video_cfg,
        checkpoint_path=Path(args.video_bridge_ckpt),
        device=device,
        dtype=dtype,
        modifier=DEFAULT_PROMPT_MODIFIER,
    )
    backend_started = time.perf_counter()
    backend = build_generation_backend(video_cfg.backend, device=device)
    backend_load_seconds = time.perf_counter() - backend_started

    rows: list[dict[str, object]] = []
    for index, (prompt, first_frame, condition) in enumerate(
        zip(prompts, first_frames, video_conditions)
    ):
        prompt_embeds, prompt_mask, pooled = (
            value.to(device=device) for value in condition
        )
        row: dict[str, object] = {
            "index": index + 1,
            "prompt": prompt,
            "seed": args.seed + index,
            "first_frame_seconds": first_frame_seconds[index],
            "first_frame": str(
                image_dir / f"{index + 1:02d}_{safe_stem(prompt)}_image_bridge.png"
            ),
        }

        if not args.skip_native_first_frame:
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            started = time.perf_counter()
            native_frames = backend.generate_video_from_bridge_condition(
                prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            )
            native_seconds = time.perf_counter() - started
            native_path = native_video_dir / f"{index + 1:02d}_{safe_stem(prompt)}.mp4"
            export_to_video(native_frames, native_path, fps=args.fps)
            row.update(
                {
                    "native_first_frame_video": str(native_path),
                    "native_first_frame_video_seconds": native_seconds,
                }
            )

        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        started = time.perf_counter()
        combined_frames = backend.generate_video_from_bridge_condition(
            prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            pooled_prompt_embeds=pooled,
            first_frame=first_frame,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        combined_seconds = time.perf_counter() - started
        combined_path = combined_video_dir / f"{index + 1:02d}_{safe_stem(prompt)}.mp4"
        export_to_video(combined_frames, combined_path, fps=args.fps)
        row.update(
            {
                "image_bridge_first_frame_video": str(combined_path),
                "image_bridge_first_frame_video_seconds": combined_seconds,
            }
        )
        rows.append(row)
        print(
            f"[{index + 1}/{len(prompts)}] {prompt} "
            f"image={first_frame_seconds[index]:.2f}s video={combined_seconds:.2f}s",
            flush=True,
        )

    summary = {
        "status": "ok",
        "video_checkpoint": video_metadata,
        "image_checkpoint": image_metadata,
        "num_prompts": len(prompts),
        "dtype": str(dtype),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "image_append_prompt_modifier": args.image_append_prompt_modifier,
        "video_append_prompt_modifier": True,
        "backend_load_seconds": backend_load_seconds,
        "prompts": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
