#!/usr/bin/env python
"""Benchmark native NeoDragon T2V against the complete Mobile-OV pipeline.

The two warm paths use identical NeoDragon Hybrid settings (512x320, 49 frames,
1-1-1 sampling).  Native uses its released SSD1B first-frame/text stack.
Mobile-OV uses a DreamLite image bridge/first-frame generator and the Exp1 text
bridge, then injects both into the same released NeoDragon Hybrid DiT.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge, MobileOVNeodragonTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend
from new_mobile_ov.generation.backends import DreamLiteMobileBackend
from new_mobile_ov.generation.quicksrnet import load_quicksrnet, model_parameter_count
from tools.apply_quicksrnet_video import upscale_frames


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name).lower() in {"bf16", "bfloat16"} else torch.float16


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - started


def peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def load_state(path: str | Path, key: str = "bridge") -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get(key, payload)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dict under {key!r}")
    return state, int(payload.get("step", -1))


def frames_to_numpy(frames) -> np.ndarray:
    return np.stack([np.asarray(frame.convert("RGB")) for frame in frames])


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def summary(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    result: dict[str, float] = {"runs": len(rows)}
    for key in keys:
        values = [row[key] for row in rows]
        result[f"mean_{key}"] = statistics.mean(values)
        result[f"median_{key}"] = statistics.median(values)
    return result


def native_hybrid_once(backend, prompt: str, *, height: int, width: int, frames: int, device: torch.device):
    """Split released native Hybrid generation at its SSD1B first-frame boundary."""

    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER, generate

    pipeline = backend.pipeline
    with torch.autocast("cuda", dtype=backend.dtype):
        first_frame, first_frame_seconds = timed(
            device,
            lambda: pipeline.first_frame_gen_pipeline(
                prompt=str(prompt) + DEFAULT_PROMPT_MODIFIER,
                num_images_per_prompt=1,
            ).images[0],
        )
        video, video_seconds = timed(
            device,
            lambda: generate(
                text_encoder_bundle=pipeline.text_encoder_bundle,
                dit=pipeline.dit,
                context_adapter=pipeline.context_adapter,
                vae=pipeline.vae,
                scheduler=pipeline.scheduler,
                prompt=str(prompt),
                image=first_frame,
                height=height,
                width=width,
                num_frames=frames,
                frames_per_unit=pipeline.config.frames_per_unit,
                num_stages=len(pipeline.config.stages),
                output_type="pil",
                device=device,
                dtype=backend.dtype,
                **pipeline.config.gen_confs[backend.mode],
            ),
        )
    return video, {"first_frame_seconds": first_frame_seconds, "video_seconds": video_seconds}


def mobile_once(
    *,
    prompt: str,
    image_bridge: MobileOVDreamLiteImageBridge,
    image_backend: DreamLiteMobileBackend,
    video_bridge: MobileOVNeodragonTextBridge,
    video_backend,
    anchor_height: int,
    anchor_width: int,
    anchor_time_id_height: int,
    anchor_time_id_width: int,
    image_steps: int,
    video_height: int,
    video_width: int,
    frames: int,
    seed: int,
    device: torch.device,
):
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    with torch.autocast("cuda", dtype=video_backend.dtype):
        condition, image_bridge_seconds = timed(device, lambda: image_bridge([prompt], mode="generate"))
        first_frame, dreamlite_seconds = timed(
            device,
            lambda: image_backend.generate_images(
                condition,
                height=anchor_height,
                width=anchor_width,
                time_id_height=anchor_time_id_height,
                time_id_width=anchor_time_id_width,
                num_steps=image_steps,
                seed=seed,
            )[0].convert("RGB"),
        )
        bridge_outputs, video_bridge_seconds = timed(
            device,
            lambda: video_bridge.encode([str(prompt) + DEFAULT_PROMPT_MODIFIER]),
        )
        prompt_embeds, prompt_mask, pooled = bridge_outputs
        video, video_seconds = timed(
            device,
            lambda: video_backend.generate_video_from_bridge_condition(
                prompt=str(prompt),
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                first_frame=first_frame,
                height=video_height,
                width=video_width,
                num_frames=frames,
            ),
        )
    return video, {
        "image_bridge_seconds": image_bridge_seconds,
        "dreamlite_first_frame_seconds": dreamlite_seconds,
        "video_bridge_seconds": video_bridge_seconds,
        "video_seconds": video_seconds,
    }


def add_qsr_timing(
    frames,
    *,
    quicksrnet: torch.nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
    batch_frames: int,
) -> dict[str, float]:
    if quicksrnet is None:
        return {"qsr_upload_seconds": 0.0, "qsr_forward_seconds": 0.0}
    _, timing = upscale_frames(
        frames_to_numpy(frames),
        model=quicksrnet,
        device=device,
        dtype=dtype,
        batch_frames=batch_frames,
        return_frames=False,
        reset_peak_memory=False,
    )
    return {
        "qsr_upload_seconds": timing["upload_seconds"],
        "qsr_forward_seconds": timing["forward_seconds"],
    }


def prune_native_only_components(video_backend) -> None:
    """The bridge path never calls these native-only modules after this point."""

    pipeline = video_backend.pipeline
    pipeline.first_frame_gen_pipeline = None
    pipeline.text_encoder_bundle = None
    pipeline.context_adapter = None
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v7.yaml")
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--image-bridge-ckpt", required=True)
    parser.add_argument("--video-bridge-ckpt", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--anchor-height", type=int, default=640)
    parser.add_argument("--anchor-width", type=int, default=1024)
    parser.add_argument("--anchor-time-id-height", type=int, default=800)
    parser.add_argument("--anchor-time-id-width", type=int, default=1280)
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--with-quicksr", action="store_true")
    parser.add_argument("--quicksr-variant", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--qsr-batch-frames", type=int, default=4)
    parser.add_argument("--output-dir", default="output/full_pipeline_latency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full pipeline latency measurement requires one CUDA GPU.")
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive.")
    prompts = args.prompt or ["A red fox walking through gentle snowfall, cinematic wildlife footage."]
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_cfg = load_config(args.image_config)
    video_cfg = load_config(args.video_config)
    dtype = dtype_from_name(video_cfg.backend.dtype)
    image_state, image_step = load_state(args.image_bridge_ckpt)
    video_state, video_step = load_state(args.video_bridge_ckpt)

    started = time.perf_counter()
    video_backend = build_generation_backend(video_cfg.backend, device=device)
    synchronize(device)
    video_backend_load_seconds = time.perf_counter() - started
    print(
        f"Loaded NeoDragon Hybrid pipeline in {video_backend_load_seconds:.2f}s; "
        f"native and Mobile-OV use the same released DiT/VAE/scheduler.",
        flush=True,
    )

    qsr = None
    qsr_checkpoint = None
    if args.with_quicksr:
        qsr, qsr_checkpoint = load_quicksrnet(
            variant=args.quicksr_variant,
            scale=2,
            cache_dir=ROOT / "checkpoints" / "quicksrnet",
            device=device,
            dtype=torch.float16,
        )

    native_rows: list[dict[str, float]] = []
    for index in range(args.warmup + args.runs):
        prompt = prompts[index % len(prompts)]
        set_seed(args.seed + index, device)
        reset_peak(device)
        frames, timing = native_hybrid_once(
            video_backend,
            prompt,
            height=args.height,
            width=args.width,
            frames=args.num_frames,
            device=device,
        )
        timing.update(
            add_qsr_timing(
                frames,
                quicksrnet=qsr,
                device=device,
                dtype=torch.float16,
                batch_frames=args.qsr_batch_frames,
            )
        )
        timing["total_seconds"] = sum(timing.values())
        timing.update(peak_memory(device))
        if index >= args.warmup:
            native_rows.append(timing)
        print(f"native {'warmup' if index < args.warmup else 'run'} {index}: {timing['total_seconds']:.3f}s", flush=True)

    prune_native_only_components(video_backend)
    started = time.perf_counter()
    image_bridge = MobileOVDreamLiteImageBridge(
        image_cfg.bridge,
        image_cfg.dreamlite_bridge,
        device=device,
        dtype=dtype,
    ).eval()
    image_bridge.load_trainable_state_dict(image_state)
    image_backend = DreamLiteMobileBackend(image_cfg.dreamlite, device=device, dtype=dtype, load_vae=True)
    video_bridge = MobileOVNeodragonTextBridge(video_cfg.bridge, device=device, dtype=dtype).eval()
    missing, unexpected = video_bridge.load_state_dict(video_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Video bridge checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del image_state, video_state
    synchronize(device)
    mobile_extra_load_seconds = time.perf_counter() - started
    print(f"Loaded Mobile-OV conditioning modules in {mobile_extra_load_seconds:.2f}s", flush=True)

    mobile_rows: list[dict[str, float]] = []
    for index in range(args.warmup + args.runs):
        prompt = prompts[index % len(prompts)]
        set_seed(args.seed + index, device)
        reset_peak(device)
        frames, timing = mobile_once(
            prompt=prompt,
            image_bridge=image_bridge,
            image_backend=image_backend,
            video_bridge=video_bridge,
            video_backend=video_backend,
            anchor_height=args.anchor_height,
            anchor_width=args.anchor_width,
            anchor_time_id_height=args.anchor_time_id_height,
            anchor_time_id_width=args.anchor_time_id_width,
            image_steps=args.image_steps,
            video_height=args.height,
            video_width=args.width,
            frames=args.num_frames,
            seed=args.seed + index,
            device=device,
        )
        timing.update(
            add_qsr_timing(
                frames,
                quicksrnet=qsr,
                device=device,
                dtype=torch.float16,
                batch_frames=args.qsr_batch_frames,
            )
        )
        timing["total_seconds"] = sum(timing.values())
        timing.update(peak_memory(device))
        if index >= args.warmup:
            mobile_rows.append(timing)
        print(f"mobile {'warmup' if index < args.warmup else 'run'} {index}: {timing['total_seconds']:.3f}s", flush=True)

    native_keys = ["first_frame_seconds", "video_seconds", "qsr_upload_seconds", "qsr_forward_seconds", "total_seconds"]
    mobile_keys = [
        "image_bridge_seconds",
        "dreamlite_first_frame_seconds",
        "video_bridge_seconds",
        "video_seconds",
        "qsr_upload_seconds",
        "qsr_forward_seconds",
        "total_seconds",
    ]
    native_summary = summary(native_rows, native_keys)
    mobile_summary = summary(mobile_rows, mobile_keys)
    payload = {
        "protocol": {
            "native": "released NeoDragon Hybrid: native SSD1B + native text stack + released Hybrid DiT/VAE",
            "mobile_ov": "DreamLite image bridge + DreamLite first frame + Exp1 text bridge + same released Hybrid DiT/VAE",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "hybrid_schedule": "1-1-1",
            "image_steps": args.image_steps,
            "warmup": args.warmup,
            "runs": args.runs,
            "quicksr": {
                "enabled": args.with_quicksr,
                "implementation": "public Qualcomm QuickSRNet reproduction" if qsr else None,
                "variant": args.quicksr_variant if qsr else None,
                "parameters": model_parameter_count(qsr) if qsr else 0,
                "checkpoint": str(qsr_checkpoint) if qsr_checkpoint else None,
            },
            "note": "Timing excludes MP4 encoding. QuickSR timing includes host-to-GPU upload plus forward, but not output download.",
        },
        "checkpoints": {
            "image_bridge": str(Path(args.image_bridge_ckpt).resolve()),
            "image_bridge_step": image_step,
            "video_bridge": str(Path(args.video_bridge_ckpt).resolve()),
            "video_bridge_step": video_step,
        },
        "load_seconds": {
            "neodragon_bundle": video_backend_load_seconds,
            "mobile_ov_extra_modules": mobile_extra_load_seconds,
            "note": "Cold loading is implementation-dependent; compare warm execution summaries below.",
        },
        "native_neodragon": {"rows": native_rows, "summary": native_summary},
        "mobile_ov": {"rows": mobile_rows, "summary": mobile_summary},
        "comparison": {
            "native_over_mobile_total_ratio": native_summary["mean_total_seconds"] / mobile_summary["mean_total_seconds"],
            "mobile_minus_native_total_seconds": mobile_summary["mean_total_seconds"] - native_summary["mean_total_seconds"],
        },
    }
    destination = output_dir / "summary.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"native": native_summary, "mobile_ov": mobile_summary, "comparison": payload["comparison"]}, indent=2), flush=True)
    print(f"Saved full-pipeline timing report: {destination}", flush=True)


if __name__ == "__main__":
    main()
