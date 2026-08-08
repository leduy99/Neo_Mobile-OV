#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
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
from new_mobile_ov.checkpoints import ensure_neodragon_assets  # noqa: E402
from new_mobile_ov.config import load_config  # noqa: E402
from new_mobile_ov.generation import build_generation_backend  # noqa: E402
from new_mobile_ov.generation.backends import DreamLiteMobileBackend  # noqa: E402


def load_prompts(info_path: Path, max_prompts: int) -> list[str]:
    rows = json.loads(info_path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = " ".join(str(row["prompt_en"]).strip().split())
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if max_prompts > 0 and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise RuntimeError(f"No prompts found in {info_path}")
    return prompts


def anchor_path(anchor_dir: Path, index: int, prompt: str) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return anchor_dir / f"{index:04d}_{digest}.png"


def video_path(video_dir: Path, prompt: str) -> Path:
    return video_dir / f"{prompt}-0.mp4"


def valid_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size
    except Exception:
        return False


def valid_video(path: Path, expected_frames: int) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        reader = VideoReader(str(path), num_threads=1)
        return len(reader) == expected_frames
    except Exception:
        return False


def load_bridge_state(path: Path, key: str = "bridge") -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get(key, payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a state dictionary under {key!r}")
    return state, int(payload.get("step", -1))


def release(*objects) -> None:
    for value in objects:
        del value
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_anchors(args, prompts: list[str], device: torch.device, dtype: torch.dtype) -> dict:
    anchor_dir = Path(args.anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    expected_size = (args.anchor_width, args.anchor_height)
    missing = [
        (index, prompt)
        for index, prompt in enumerate(prompts)
        if not valid_image(anchor_path(anchor_dir, index, prompt), expected_size)
    ]
    if not missing:
        print(f"Anchor phase already complete: {len(prompts)}/{len(prompts)}", flush=True)
        return {"generated": 0, "reused": len(prompts)}

    config = load_config(args.image_config)
    state, checkpoint_step = load_bridge_state(Path(args.image_bridge_checkpoint))
    bridge = MobileOVDreamLiteImageBridge(
        config.bridge,
        config.dreamlite_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    bridge.load_trainable_state_dict(state)
    backend = DreamLiteMobileBackend(config.dreamlite, device=device, dtype=dtype, load_vae=True)
    del state
    generated = 0
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=dtype):
        for index, prompt in missing:
            condition = bridge([prompt], mode="generate")
            image = backend.generate_images(
                condition,
                height=args.anchor_height,
                width=args.anchor_width,
                time_id_height=args.anchor_time_id_height,
                time_id_width=args.anchor_time_id_width,
                num_steps=args.image_steps,
                seed=args.seed + index,
            )[0].convert("RGB")
            destination = anchor_path(anchor_dir, index, prompt)
            temporary = destination.with_suffix(".tmp.png")
            image.save(temporary)
            os.replace(temporary, destination)
            generated += 1
            if generated == 1 or generated % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Anchors {len(prompts) - len(missing) + generated}/{len(prompts)} "
                    f"new={generated} rate={generated / max(elapsed, 1e-6):.2f}/s",
                    flush=True,
                )
    release(backend, bridge)
    return {
        "checkpoint_step": checkpoint_step,
        "generated": generated,
        "reused": len(prompts) - generated,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def generate_videos(args, prompts: list[str], device: torch.device, dtype: torch.dtype) -> dict:
    anchor_dir = Path(args.anchor_dir)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        (index, prompt)
        for index, prompt in enumerate(prompts)
        if not valid_video(video_path(video_dir, prompt), args.num_frames)
    ]
    if not missing:
        print(f"Video phase already complete: {len(prompts)}/{len(prompts)}", flush=True)
        return {"generated": 0, "reused": len(prompts)}

    config = load_config(args.video_config)
    repo_path, _, _ = ensure_neodragon_assets(
        repo_path=config.backend.extra.get("repo_path"),
        cache_dir=config.backend.extra.get("cache_dir"),
        model_id=config.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=config.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from new_mobile_ov.generation.neodragon_compat import (
        install_neodragon_generation_patches,
    )

    install_neodragon_generation_patches()
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    state, checkpoint_step = load_bridge_state(Path(args.video_bridge_checkpoint))
    bridge = MobileOVNeodragonTextBridge(config.bridge, device=device, dtype=dtype).eval()
    missing_keys, unexpected_keys = bridge.load_state_dict(state, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Exp1 checkpoint does not match MobileOVNeodragonTextBridge: "
            f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )
    del state
    backend = build_generation_backend(config.backend, device=device)
    generated = 0
    started = time.perf_counter()
    for index, prompt in missing:
        conditioned_prompt = prompt + DEFAULT_PROMPT_MODIFIER
        prompt_embeds, prompt_mask, pooled = bridge.encode([conditioned_prompt])
        first_frame = Image.open(anchor_path(anchor_dir, index, prompt)).convert("RGB")
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        with torch.autocast("cuda", dtype=dtype):
            frames = backend.generate_video_from_bridge_condition(
                prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                first_frame=first_frame,
                height=args.video_height,
                width=args.video_width,
                num_frames=args.num_frames,
            )
        destination = video_path(video_dir, prompt)
        temporary = destination.with_suffix(".tmp.mp4")
        export_to_video(frames, temporary, fps=args.fps)
        os.replace(temporary, destination)
        generated += 1
        if generated == 1 or generated % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Videos {len(prompts) - len(missing) + generated}/{len(prompts)} "
                f"new={generated} rate={generated / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )
    release(backend, bridge)
    return {
        "checkpoint_step": checkpoint_step,
        "generated": generated,
        "reused": len(prompts) - generated,
        "seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe VBench generation with DreamLite anchors and Exp1-64K video conditioning."
    )
    parser.add_argument("--vbench-info", required=True)
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v4.yaml")
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--image-bridge-checkpoint", required=True)
    parser.add_argument("--video-bridge-checkpoint", required=True)
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--anchor-width", type=int, default=1024)
    parser.add_argument("--anchor-height", type=int, default=640)
    parser.add_argument("--anchor-time-id-width", type=int, default=1280)
    parser.add_argument("--anchor-time-id-height", type=int, default=800)
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VBench generation requires one allocated CUDA GPU.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompts = load_prompts(Path(args.vbench_info), args.max_prompts)
    print(
        f"VBench generation prompts={len(prompts)} anchor={args.anchor_width}x{args.anchor_height}"
        f"@{args.anchor_time_id_width}x{args.anchor_time_id_height} "
        f"video={args.video_width}x{args.video_height}x{args.num_frames}",
        flush=True,
    )
    started = time.perf_counter()
    anchor_summary = generate_anchors(args, prompts, device, dtype)
    video_summary = generate_videos(args, prompts, device, dtype)
    missing_anchors = [
        prompt
        for index, prompt in enumerate(prompts)
        if not valid_image(
            anchor_path(Path(args.anchor_dir), index, prompt),
            (args.anchor_width, args.anchor_height),
        )
    ]
    missing_videos = [
        prompt
        for prompt in prompts
        if not valid_video(video_path(Path(args.video_dir), prompt), args.num_frames)
    ]
    summary = {
        "status": "ok" if not missing_anchors and not missing_videos else "incomplete",
        "unique_prompts": len(prompts),
        "image_checkpoint": str(Path(args.image_bridge_checkpoint).resolve()),
        "video_checkpoint": str(Path(args.video_bridge_checkpoint).resolve()),
        "anchor": {
            "width": args.anchor_width,
            "height": args.anchor_height,
            "time_id_width": args.anchor_time_id_width,
            "time_id_height": args.anchor_time_id_height,
            **anchor_summary,
        },
        "video": {
            "width": args.video_width,
            "height": args.video_height,
            "frames": args.num_frames,
            "fps": args.fps,
            **video_summary,
        },
        "missing_anchors": missing_anchors,
        "missing_videos": missing_videos,
        "total_seconds": time.perf_counter() - started,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "ok":
        raise SystemExit("Generation did not complete; rerun the same command to resume.")


if __name__ == "__main__":
    main()
