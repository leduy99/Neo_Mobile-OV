#!/usr/bin/env python
"""Apply or benchmark public QuickSRNet on already-generated MP4 files.

This intentionally does not invoke DreamLite, a bridge, or NeoDragon.  It is a
post-process control: feed the same existing videos from two pipelines through
the same QSR model before comparing image-quality metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.generation.quicksrnet import load_quicksrnet, model_parameter_count


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    choices = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return choices[name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.input_video]
    if args.input_dir:
        paths.extend(Path(args.input_dir).glob(args.glob))
    unique = sorted({path.resolve() for path in paths})
    valid = [path for path in unique if path.is_file()]
    if not valid:
        raise FileNotFoundError("No input videos found. Set --input-video or --input-dir with a matching --glob.")
    if args.limit:
        valid = valid[: args.limit]
    return valid


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames), fps


@torch.inference_mode()
def upscale_frames(
    frames: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    batch_frames: int,
    return_frames: bool = True,
) -> tuple[np.ndarray | None, dict[str, float]]:
    outputs: list[np.ndarray] = []
    upload_seconds = 0.0
    forward_seconds = 0.0
    download_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for offset in range(0, len(frames), batch_frames):
        batch = frames[offset : offset + batch_frames]
        cpu_input = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().div_(255.0)

        synchronize(device)
        started = time.perf_counter()
        inputs = cpu_input.to(device=device, dtype=dtype, non_blocking=True)
        synchronize(device)
        upload_seconds += time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        predictions = model(inputs)
        synchronize(device)
        forward_seconds += time.perf_counter() - started

        if return_frames:
            synchronize(device)
            started = time.perf_counter()
            rgb = predictions.float().clamp_(0.0, 1.0).mul_(255.0).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            synchronize(device)
            download_seconds += time.perf_counter() - started
            outputs.append(rgb)
        del predictions, inputs, cpu_input

    return (np.concatenate(outputs, axis=0) if return_frames else None), {
        "upload_seconds": upload_seconds,
        "forward_seconds": forward_seconds,
        "download_seconds": download_seconds,
        **cuda_memory(device),
    }


def write_video(path: Path, frames_rgb: np.ndarray, fps: float, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    temporary = path.with_suffix(".partial.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temporary}; codec={codec!r}")
    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    os.replace(temporary, path)


def summarize_benchmark(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "runs": len(rows),
        "mean_clip_forward_seconds": statistics.mean(row["forward_seconds"] for row in rows),
        "median_clip_forward_seconds": statistics.median(row["forward_seconds"] for row in rows),
        "mean_ms_per_frame": statistics.mean(1000.0 * row["forward_seconds"] / row["frames"] for row in rows),
        "median_ms_per_frame": statistics.median(1000.0 * row["forward_seconds"] / row["frames"] for row in rows),
        "mean_peak_alloc_gb": statistics.mean(row.get("peak_alloc_gb", 0.0) for row in rows),
        "mean_peak_reserved_gb": statistics.mean(row.get("peak_reserved_gb", 0.0) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", action="append", default=[], help="MP4 input; may be repeated.")
    parser.add_argument("--input-dir", help="Recursively select existing videos from this directory.")
    parser.add_argument("--glob", default="**/*.mp4", help="Pattern used with --input-dir.")
    parser.add_argument("--output-dir", default="output/quicksrnet_video_postprocess")
    parser.add_argument("--variant", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--checkpoint", help="Optional existing public QuickSRNet checkpoint.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/quicksrnet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--codec", default="mp4v", help="FourCC used by OpenCV's MP4 writer.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on selected videos.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=0, help="GPU-only repeats on the first decoded clip; no video is written.")
    parser.add_argument("--warmup-runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_frames < 1:
        raise ValueError("--batch-frames must be positive")
    if args.benchmark_runs < 0 or args.warmup_runs < 0:
        raise ValueError("--benchmark-runs and --warmup-runs must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    dtype = resolve_dtype(args.dtype, device)
    inputs = collect_inputs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model, checkpoint = load_quicksrnet(
        variant=args.variant,
        scale=args.scale,
        checkpoint=args.checkpoint,
        cache_dir=args.checkpoint_dir,
        device=device,
        dtype=dtype,
    )
    synchronize(device)
    load_seconds = time.perf_counter() - started
    model_info = {
        "implementation": "public_qualcomm_quicksrnet_reproduction",
        "variant": args.variant,
        "scale": args.scale,
        "checkpoint": str(checkpoint),
        "parameter_count": model_parameter_count(model),
        "device": str(device),
        "dtype": str(dtype),
        "model_load_seconds": load_seconds,
    }
    print(json.dumps({"model": model_info, "selected_videos": len(inputs)}, indent=2), flush=True)

    if args.benchmark_runs:
        frames, fps = read_video(inputs[0])
        for _ in range(args.warmup_runs):
            upscale_frames(
                frames,
                model=model,
                device=device,
                dtype=dtype,
                batch_frames=args.batch_frames,
                return_frames=False,
            )
        rows: list[dict[str, float]] = []
        for run in range(args.benchmark_runs):
            _, timing = upscale_frames(
                frames,
                model=model,
                device=device,
                dtype=dtype,
                batch_frames=args.batch_frames,
                return_frames=False,
            )
            rows.append({"run": run, "frames": float(len(frames)), "fps": fps, **timing})
        payload = {
            "mode": "gpu_forward_only",
            "input_video": str(inputs[0]),
            "input_shape": list(frames.shape),
            "model": model_info,
            "rows": rows,
            "summary": summarize_benchmark(rows),
            "note": "GPU forward timing excludes video decode and MP4 writing.",
        }
        destination = output_dir / "quicksrnet_gpu_benchmark.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "mode": payload["mode"],
                    "input_shape": payload["input_shape"],
                    "summary": payload["summary"],
                    "full_report": str(destination),
                },
                indent=2,
            ),
            flush=True,
        )
        print(f"Saved benchmark: {destination}", flush=True)
        return

    report_rows: list[dict[str, object]] = []
    for index, input_path in enumerate(inputs, start=1):
        relative = input_path.name if not args.input_dir else input_path.relative_to(Path(args.input_dir).resolve())
        output_path = output_dir / relative
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_path}", flush=True)
            continue

        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        frames, fps = read_video(input_path)
        decode_seconds = time.perf_counter() - decode_started
        upscaled, timing = upscale_frames(
            frames,
            model=model,
            device=device,
            dtype=dtype,
            batch_frames=args.batch_frames,
        )
        assert upscaled is not None
        write_started = time.perf_counter()
        write_video(output_path, upscaled, fps, args.codec)
        write_seconds = time.perf_counter() - write_started
        total_seconds = time.perf_counter() - total_started
        row: dict[str, object] = {
            "input_video": str(input_path),
            "output_video": str(output_path),
            "frames": int(len(frames)),
            "input_height": int(frames.shape[1]),
            "input_width": int(frames.shape[2]),
            "output_height": int(upscaled.shape[1]),
            "output_width": int(upscaled.shape[2]),
            "fps": fps,
            "decode_seconds": decode_seconds,
            "write_seconds": write_seconds,
            "total_seconds": total_seconds,
            "end_to_end_ms_per_frame": 1000.0 * total_seconds / len(frames),
            "gpu_forward_ms_per_frame": 1000.0 * timing["forward_seconds"] / len(frames),
            **timing,
        }
        report_rows.append(row)
        print(
            f"[{index}/{len(inputs)}] {input_path.name}: {frames.shape[2]}x{frames.shape[1]} -> "
            f"{upscaled.shape[2]}x{upscaled.shape[1]}, gpu={row['gpu_forward_ms_per_frame']:.2f} ms/frame, "
            f"e2e={row['end_to_end_ms_per_frame']:.2f} ms/frame",
            flush=True,
        )

    payload = {
        "mode": "apply_existing_videos",
        "model": model_info,
        "rows": report_rows,
        "note": "Each input was an existing generated MP4; no image or video generator was invoked.",
    }
    destination = output_dir / "quicksrnet_apply_report.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report: {destination}", flush=True)


if __name__ == "__main__":
    main()
