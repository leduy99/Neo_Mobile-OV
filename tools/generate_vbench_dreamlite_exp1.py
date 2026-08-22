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

import numpy as np
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


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).strip().split())


def load_prompts(info_path: Path, max_prompts: int) -> list[str]:
    rows = json.loads(info_path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = normalize_prompt(row["prompt_en"])
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if max_prompts > 0 and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise RuntimeError(f"No prompts found in {info_path}")
    return prompts


def load_conditioning_prompts(recaption_file: Path | None, prompts: list[str]) -> list[str]:
    """Return bridge inputs while retaining raw VBench prompts for filenames/scoring."""

    if recaption_file is None:
        return list(prompts)
    payload = json.loads(recaption_file.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid VBench recaption file: {recaption_file}")
    mapping: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = normalize_prompt(row.get("prompt", ""))
        recaption = normalize_prompt(row.get("recaption", ""))
        if raw and recaption:
            mapping[raw] = recaption
    missing = [prompt for prompt in prompts if prompt not in mapping]
    if missing:
        raise RuntimeError(
            f"Recaption file {recaption_file} is missing {len(missing)} VBench prompt(s); "
            f"first={missing[0]!r}"
        )
    return [mapping[prompt] for prompt in prompts]


def anchor_path(
    anchor_dir: Path,
    index: int,
    prompt: str,
    sample_index: int = 0,
    samples_per_prompt: int = 1,
) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    if samples_per_prompt > 1:
        return anchor_dir / f"{index:04d}_{digest}-{sample_index}.png"
    return anchor_dir / f"{index:04d}_{digest}.png"


def video_path(video_dir: Path, prompt: str, sample_index: int = 0) -> Path:
    return video_dir / f"{prompt}-{sample_index}.mp4"


def sample_seed(
    base_seed: int,
    prompt_index: int,
    sample_index: int,
    samples_per_prompt: int,
) -> int:
    return int(base_seed) + int(prompt_index) * int(samples_per_prompt) + int(sample_index)


def sample_items(prompts: list[str], samples_per_prompt: int):
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(samples_per_prompt):
            yield prompt_index, prompt, sample_index


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


def valid_video(
    path: Path,
    expected_frames: int,
    expected_size: tuple[int, int] | None = None,
) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        reader = VideoReader(str(path), num_threads=1)
        if len(reader) != expected_frames:
            return False
        if expected_size is None:
            return True
        frame = reader[0]
        height, width = frame.shape[:2]
        return (width, height) == expected_size
    except Exception:
        return False


def output_video_size(args: argparse.Namespace) -> tuple[int, int]:
    scale = args.quicksr_scale if args.with_quicksr else 1
    return args.video_width * scale, args.video_height * scale


def quicksr_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


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
def generate_anchors(
    args,
    prompts: list[str],
    conditioning_prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    anchor_dir = Path(args.anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    expected_size = (args.anchor_width, args.anchor_height)
    total = len(prompts) * args.samples_per_prompt
    missing = [
        (prompt_index, prompt, sample_index)
        for prompt_index, prompt, sample_index in sample_items(
            prompts,
            args.samples_per_prompt,
        )
        if not valid_image(
            anchor_path(
                anchor_dir,
                prompt_index,
                prompt,
                sample_index,
                args.samples_per_prompt,
            ),
            expected_size,
        )
    ]
    if not missing:
        print(f"Anchor phase already complete: {total}/{total}", flush=True)
        return {"generated": 0, "reused": total}

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
        for prompt_index, prompt, sample_index in missing:
            condition = bridge([conditioning_prompts[prompt_index]], mode="generate")
            image = backend.generate_images(
                condition,
                height=args.anchor_height,
                width=args.anchor_width,
                time_id_height=args.anchor_time_id_height,
                time_id_width=args.anchor_time_id_width,
                num_steps=args.image_steps,
                seed=sample_seed(
                    args.seed,
                    prompt_index,
                    sample_index,
                    args.samples_per_prompt,
                ),
            )[0].convert("RGB")
            destination = anchor_path(
                anchor_dir,
                prompt_index,
                prompt,
                sample_index,
                args.samples_per_prompt,
            )
            temporary = destination.with_suffix(".tmp.png")
            image.save(temporary)
            os.replace(temporary, destination)
            generated += 1
            if generated == 1 or generated % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Anchors {total - len(missing) + generated}/{total} "
                    f"new={generated} rate={generated / max(elapsed, 1e-6):.2f}/s",
                    flush=True,
                )
    release(backend, bridge)
    return {
        "checkpoint_step": checkpoint_step,
        "generated": generated,
        "reused": total - generated,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def generate_videos(
    args,
    prompts: list[str],
    conditioning_prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    anchor_dir = Path(args.anchor_dir)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    total = len(prompts) * args.samples_per_prompt
    missing = [
        (prompt_index, prompt, sample_index)
        for prompt_index, prompt, sample_index in sample_items(
            prompts,
            args.samples_per_prompt,
        )
        if not valid_video(
            video_path(video_dir, prompt, sample_index),
            args.num_frames,
            output_video_size(args),
        )
    ]
    if not missing:
        print(f"Video phase already complete: {total}/{total}", flush=True)
        return {"generated": 0, "reused": total}

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

    joint_checkpoint = (
        None
        if args.joint_video_checkpoint is None
        else Path(args.joint_video_checkpoint).expanduser().resolve()
    )
    joint_payload = None
    if joint_checkpoint is not None:
        joint_payload = torch.load(
            joint_checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        stack = joint_payload.get("teacher_stack")
        if not isinstance(stack, dict) or stack.get("name") != "multistep":
            raise RuntimeError(
                f"Joint checkpoint is not a multistep monolithic model: {joint_checkpoint}"
            )
        if not isinstance(joint_payload.get("dit"), dict) or not isinstance(
            joint_payload.get("bridge"), dict
        ):
            raise RuntimeError(
                f"Joint checkpoint must contain both 'dit' and 'bridge': {joint_checkpoint}"
            )
        state = joint_payload["bridge"]
        checkpoint_step = int(joint_payload.get("step", -1))
        config.backend.extra["mode"] = "monolithic"
    else:
        state, checkpoint_step = load_bridge_state(Path(args.video_bridge_checkpoint))

    bridge = MobileOVNeodragonTextBridge(config.bridge, device=device, dtype=dtype).eval()
    missing_keys, unexpected_keys = bridge.load_state_dict(
        state,
        strict=joint_payload is not None,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Exp1 checkpoint does not match MobileOVNeodragonTextBridge: "
            f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )
    backend = build_generation_backend(config.backend, device=device)
    if joint_payload is not None:
        backend.pipeline.dit.load_state_dict(joint_payload["dit"], strict=True)
        # Bridge inference bypasses both native text modules. Releasing them keeps
        # the full 944-prompt run comfortably below one-GPU memory limits.
        backend.pipeline.text_encoder_bundle = None
        backend.pipeline.context_adapter = None
        del joint_payload
    del state
    gc.collect()
    torch.cuda.empty_cache()

    negative_condition = None
    if joint_checkpoint is not None:
        from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT

        negative_condition = bridge.encode([DEFAULT_NEGATIVE_PROMPT])
    quicksr_model = None
    quicksr_summary: dict[str, object] = {"enabled": bool(args.with_quicksr)}
    if args.with_quicksr:
        from new_mobile_ov.generation.quicksrnet import load_quicksrnet, model_parameter_count
        from tools.apply_quicksrnet_video import upscale_frames

        quicksr_model, quicksr_checkpoint = load_quicksrnet(
            variant=args.quicksr_variant,
            scale=args.quicksr_scale,
            checkpoint=args.quicksr_checkpoint,
            cache_dir=args.quicksr_checkpoint_dir,
            device=device,
            dtype=quicksr_dtype(args.quicksr_dtype, device),
        )
        quicksr_summary.update(
            {
                "variant": args.quicksr_variant,
                "scale": args.quicksr_scale,
                "checkpoint": str(quicksr_checkpoint.resolve()),
                "parameter_count": model_parameter_count(quicksr_model),
                "dtype": args.quicksr_dtype,
                "batch_frames": args.quicksr_batch_frames,
                "upload_seconds": 0.0,
                "forward_seconds": 0.0,
                "download_seconds": 0.0,
            }
        )
    generated = 0
    started = time.perf_counter()
    for prompt_index, prompt, sample_index in missing:
        conditioned_prompt = conditioning_prompts[prompt_index] + DEFAULT_PROMPT_MODIFIER
        prompt_embeds, prompt_mask, pooled = bridge.encode([conditioned_prompt])
        first_frame = Image.open(
            anchor_path(
                anchor_dir,
                prompt_index,
                prompt,
                sample_index,
                args.samples_per_prompt,
            )
        ).convert("RGB")
        seed = sample_seed(
            args.seed,
            prompt_index,
            sample_index,
            args.samples_per_prompt,
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=dtype):
            negative_kwargs = {}
            if negative_condition is not None:
                negative_kwargs = {
                    "negative_prompt_embeds": negative_condition[0],
                    "negative_prompt_mask": negative_condition[1],
                    "negative_pooled_prompt_embeds": negative_condition[2],
                }
            frames = backend.generate_video_from_bridge_condition(
                prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                first_frame=first_frame,
                height=args.video_height,
                width=args.video_width,
                num_frames=args.num_frames,
                **negative_kwargs,
            )
        if quicksr_model is not None:
            frames_array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
            # Keep QuickSR's requested precision instead of inheriting the BF16 DiT autocast.
            with torch.autocast("cuda", enabled=False):
                upscaled, timing = upscale_frames(
                    frames_array,
                    model=quicksr_model,
                    device=device,
                    dtype=quicksr_dtype(args.quicksr_dtype, device),
                    batch_frames=args.quicksr_batch_frames,
                    reset_peak_memory=False,
                )
            if upscaled is None:
                raise RuntimeError("QuickSR did not return output frames.")
            frames = [Image.fromarray(frame) for frame in upscaled]
            for field in ("upload_seconds", "forward_seconds", "download_seconds"):
                quicksr_summary[field] = float(quicksr_summary[field]) + timing[field]
        destination = video_path(video_dir, prompt, sample_index)
        temporary = destination.with_suffix(".tmp.mp4")
        export_to_video(frames, temporary, fps=args.fps)
        os.replace(temporary, destination)
        generated += 1
        if generated == 1 or generated % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Videos {total - len(missing) + generated}/{total} "
                f"new={generated} rate={generated / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )
    release(backend, bridge, quicksr_model)
    summary = {
        "checkpoint_step": checkpoint_step,
        "generated": generated,
        "reused": total - generated,
        "seconds": time.perf_counter() - started,
    }
    if args.with_quicksr:
        summary["quicksr"] = quicksr_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe VBench generation with DreamLite anchors and Exp1-64K video conditioning."
    )
    parser.add_argument("--vbench-info", required=True)
    parser.add_argument(
        "--recaption-file",
        type=Path,
        help="Optional JSON produced by recaption_vbench_prompts_smolvlm2.py. Raw VBench prompts remain video filenames.",
    )
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v4.yaml")
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--image-bridge-checkpoint", required=True)
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument("--video-bridge-checkpoint")
    video_group.add_argument(
        "--joint-video-checkpoint",
        help=(
            "Joint multistep checkpoint containing both 'dit' and 'bridge'. "
            "Generation uses monolithic CFG and the provided DreamLite anchor."
        ),
    )
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
    parser.add_argument(
        "--with-quicksr",
        action="store_true",
        help="Apply public Qualcomm QuickSRNet to every generated video before MP4 export.",
    )
    parser.add_argument("--quicksr-variant", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--quicksr-scale", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--quicksr-checkpoint")
    parser.add_argument("--quicksr-checkpoint-dir", default="checkpoints/quicksrnet")
    parser.add_argument("--quicksr-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--quicksr-batch-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Generate independently seeded samples named PROMPT-0.mp4 through PROMPT-(N-1).mp4.",
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VBench generation requires one allocated CUDA GPU.")
    if args.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be positive")
    if args.quicksr_batch_frames <= 0:
        raise ValueError("--quicksr-batch-frames must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompts = load_prompts(Path(args.vbench_info), args.max_prompts)
    conditioning_prompts = load_conditioning_prompts(args.recaption_file, prompts)
    print(
        f"VBench generation prompts={len(prompts)} samples_per_prompt={args.samples_per_prompt} "
        f"expected_videos={len(prompts) * args.samples_per_prompt} "
        f"anchor={args.anchor_width}x{args.anchor_height}"
        f"@{args.anchor_time_id_width}x{args.anchor_time_id_height} "
        f"video={args.video_width}x{args.video_height}x{args.num_frames} "
        f"output={output_video_size(args)[0]}x{output_video_size(args)[1]} "
        f"quicksr={args.quicksr_variant}x{args.quicksr_scale if args.with_quicksr else 1} "
        f"conditioning={'smolvlm2_recaption' if args.recaption_file else 'raw_vbench'}",
        flush=True,
    )
    started = time.perf_counter()
    anchor_summary = generate_anchors(args, prompts, conditioning_prompts, device, dtype)
    video_summary = generate_videos(args, prompts, conditioning_prompts, device, dtype)
    missing_anchors = [
        {"prompt": prompt, "sample_index": sample_index}
        for index, prompt, sample_index in sample_items(prompts, args.samples_per_prompt)
        if not valid_image(
            anchor_path(
                Path(args.anchor_dir),
                index,
                prompt,
                sample_index,
                args.samples_per_prompt,
            ),
            (args.anchor_width, args.anchor_height),
        )
    ]
    missing_videos = [
        {"prompt": prompt, "sample_index": sample_index}
        for _, prompt, sample_index in sample_items(prompts, args.samples_per_prompt)
        if not valid_video(
            video_path(Path(args.video_dir), prompt, sample_index),
            args.num_frames,
            output_video_size(args),
        )
    ]
    summary = {
        "status": "ok" if not missing_anchors and not missing_videos else "incomplete",
        "unique_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "expected_anchors": len(prompts) * args.samples_per_prompt,
        "expected_videos": len(prompts) * args.samples_per_prompt,
        "image_checkpoint": str(Path(args.image_bridge_checkpoint).resolve()),
        "video_checkpoint": str(
            Path(args.joint_video_checkpoint or args.video_bridge_checkpoint).resolve()
        ),
        "video_checkpoint_kind": (
            "joint_monolithic" if args.joint_video_checkpoint else "bridge_only_hybrid"
        ),
        "recaption_file": (
            None if args.recaption_file is None else str(args.recaption_file.resolve())
        ),
        "conditioning_prompts_differ": sum(
            raw != conditioned for raw, conditioned in zip(prompts, conditioning_prompts)
        ),
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
            "output_width": output_video_size(args)[0],
            "output_height": output_video_size(args)[1],
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
