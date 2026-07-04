#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation import build_generation_backend


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def timed_call(device: torch.device, fn):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()
    result = fn()
    sync(device)
    seconds = time.perf_counter() - t0
    mem = memory_gb(device)
    return result, seconds, mem


def load_bridge(cfg, device: torch.device, dtype: torch.dtype, ckpt_path: str | None):
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("bridge", ckpt.get("student_state", ckpt))
        missing, unexpected = bridge.load_state_dict(state, strict=False)
    else:
        missing, unexpected = [], []
    return bridge, len(missing), len(unexpected)


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode in sorted({row["mode"] for row in rows}):
        cur = [row for row in rows if row["mode"] == mode]
        seconds = [float(row["total_seconds"]) for row in cur]
        out[mode] = {
            "runs": float(len(cur)),
            "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
            "mean_fps": statistics.mean(float(row["fps"]) for row in cur),
            "mean_peak_alloc_gb": statistics.mean(float(row.get("peak_alloc_gb", 0.0)) for row in cur),
            "mean_peak_reserved_gb": statistics.mean(float(row.get("peak_reserved_gb", 0.0)) for row in cur),
        }
        if mode == "mobile_ov_bridge":
            out[mode]["mean_bridge_encode_seconds"] = statistics.mean(
                float(row["bridge_encode_seconds"]) for row in cur
            )
            out[mode]["mean_video_generation_seconds"] = statistics.mean(
                float(row["video_generation_seconds"]) for row in cur
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--bridge-ckpt", default="output/neodragon_text_bridge_smoke/bridge/neodragon_text_bridge_latest.pt")
    parser.add_argument("--prompt", default="A red fox walking through gentle snowfall, cinematic wildlife footage.")
    parser.add_argument("--output-dir", default="output/neodragon_speed_benchmark")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-videos", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.backend.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_t0 = time.perf_counter()
    backend = build_generation_backend(cfg.backend, device=device)
    sync(device)
    backend_load_seconds = time.perf_counter() - load_t0

    bridge_ckpt = args.bridge_ckpt if args.bridge_ckpt and Path(args.bridge_ckpt).exists() else None
    bridge_t0 = time.perf_counter()
    bridge, missing, unexpected = load_bridge(cfg, device, dtype, bridge_ckpt)
    sync(device)
    bridge_load_seconds = time.perf_counter() - bridge_t0

    height = int(args.height or cfg.data.height)
    width = int(args.width or cfg.data.width)
    num_frames = int(args.num_frames or cfg.data.frame_num)

    metadata = {
        "config": args.config,
        "prompt": args.prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps_arg": args.fps,
        "device": str(device),
        "dtype": str(dtype),
        "backend_load_seconds": backend_load_seconds,
        "bridge_load_seconds": bridge_load_seconds,
        "bridge_ckpt": bridge_ckpt,
        "bridge_missing": missing,
        "bridge_unexpected": unexpected,
        "memory_after_load": memory_gb(device),
    }

    rows: list[dict] = []

    def set_seed(offset: int) -> None:
        torch.manual_seed(args.seed + offset)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + offset)

    total_iters = args.warmup + args.runs
    for idx in range(total_iters):
        measured = idx >= args.warmup
        run_idx = idx - args.warmup

        set_seed(idx * 2)
        native_frames, native_seconds, native_mem = timed_call(
            device,
            lambda: backend.generate_video_from_prompt(
                args.prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                profile=False,
            ),
        )
        if measured:
            row = {
                "mode": "native_neodragon",
                "run": run_idx,
                "total_seconds": native_seconds,
                "video_generation_seconds": native_seconds,
                "bridge_encode_seconds": 0.0,
                "fps": len(native_frames) / native_seconds,
                **native_mem,
            }
            rows.append(row)
            if args.save_videos and run_idx == 0:
                export_to_video(native_frames, out_dir / "native_neodragon_run0.mp4", fps=args.fps)

        set_seed(idx * 2 + 1)
        bridge_outputs, bridge_encode_seconds, bridge_mem = timed_call(
            device,
            lambda: bridge.encode([args.prompt]),
        )
        prompt_embeds, prompt_mask, pooled = bridge_outputs
        bridge_frames, bridge_video_seconds, bridge_video_mem = timed_call(
            device,
            lambda: backend.generate_video_from_bridge_condition(
                args.prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                pooled_prompt_embeds=pooled,
                height=height,
                width=width,
                num_frames=num_frames,
                profile=False,
            ),
        )
        if measured:
            total_seconds = bridge_encode_seconds + bridge_video_seconds
            row = {
                "mode": "mobile_ov_bridge",
                "run": run_idx,
                "total_seconds": total_seconds,
                "video_generation_seconds": bridge_video_seconds,
                "bridge_encode_seconds": bridge_encode_seconds,
                "fps": len(bridge_frames) / total_seconds,
                **{
                    key: max(float(bridge_mem.get(key, 0.0)), float(bridge_video_mem.get(key, 0.0)))
                    for key in set(bridge_mem) | set(bridge_video_mem)
                },
            }
            rows.append(row)
            if args.save_videos and run_idx == 0:
                export_to_video(bridge_frames, out_dir / "mobile_ov_bridge_run0.mp4", fps=args.fps)

    summary = summarize(rows)
    if "native_neodragon" in summary and "mobile_ov_bridge" in summary:
        n = summary["native_neodragon"]["mean_seconds"]
        b = summary["mobile_ov_bridge"]["mean_seconds"]
        summary["speedup_mobile_ov_vs_native"] = {"mean_seconds_ratio_native_over_bridge": n / b}

    payload = {"metadata": metadata, "rows": rows, "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "mode",
            "run",
            "total_seconds",
            "video_generation_seconds",
            "bridge_encode_seconds",
            "fps",
            "alloc_gb",
            "reserved_gb",
            "peak_alloc_gb",
            "peak_reserved_gb",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
