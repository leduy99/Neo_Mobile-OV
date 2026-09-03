#!/usr/bin/env python
# ruff: noqa: E402
"""Score OpenVid records for visual-text alignment and usable temporal motion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.training.distributed import (  # noqa: E402
    barrier,
    cleanup_distributed,
    setup_distributed,
)
from tools.score_image_bridge_grounding_siglip import (  # noqa: E402
    SiglipPairScorer,
    torch_dtype,
)


SCORE_FIELDS = (
    "video_score_status",
    "video_score_error",
    "siglip_score",
    "siglip_logit",
    "motion_frame_diff_mean",
    "motion_optical_flow_mean",
    "motion_dynamic_fraction",
    "transition_max_diff",
    "decoded_frame_count",
    "siglip_model",
    "siglip_revision",
)
TEXT_COLUMNS = (
    "caption_medium",
    "caption_long",
    "prompt",
    "caption",
    "text",
    "caption_short",
)


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def first_value(row: dict[str, str], columns: Sequence[str]) -> str:
    for column in columns:
        value = normalize_text(row.get(column, ""))
        if value:
            return value
    return ""


def record_id(row: dict[str, str], row_number: int) -> str:
    return first_value(
        row, ("record_id", "sample_id", "source_id", "index", "id")
    ) or str(row_number)


def safe_float(value: object, default: float) -> float:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else default
    except Exception:
        return default


def safe_int(value: object, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def resolve_video_path(row: dict[str, str], manifest: Path) -> Path:
    value = first_value(row, ("video_path", "media_path", "video", "path", "mp4"))
    if not value:
        raise FileNotFoundError("row has no video path")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, manifest.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def resolve_optional_path(value: str, manifest: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, manifest.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return value


def center_crop_resize(frame: np.ndarray, *, width: int) -> np.ndarray:
    height, source_width = frame.shape[:2]
    scale = width / max(source_width, 1)
    resized = cv2.resize(
        frame,
        (width, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    target_height = round(width * 9 / 16)
    if resized.shape[0] < target_height:
        scale = target_height / resized.shape[0]
        resized = cv2.resize(
            resized,
            (max(width, round(resized.shape[1] * scale)), target_height),
            interpolation=cv2.INTER_AREA,
        )
    x0 = max((resized.shape[1] - width) // 2, 0)
    y0 = max((resized.shape[0] - target_height) // 2, 0)
    return resized[y0 : y0 + target_height, x0 : x0 + width]


def motion_metrics(frames_rgb: Sequence[np.ndarray]) -> dict[str, float]:
    if len(frames_rgb) < 2:
        raise ValueError("Need at least two frames for motion scoring")
    gray = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            (192, 108),
            interpolation=cv2.INTER_AREA,
        )
        for frame in frames_rgb
    ]
    differences: list[float] = []
    flows: list[float] = []
    for previous, current in zip(gray, gray[1:]):
        differences.append(float(np.mean(np.abs(current.astype(np.float32) - previous)) / 255.0))
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(float(np.linalg.norm(flow, axis=-1).mean()))
    return {
        "motion_frame_diff_mean": float(np.mean(differences)),
        "motion_optical_flow_mean": float(np.mean(flows)),
        "motion_dynamic_fraction": float(np.mean(np.asarray(differences) >= 0.01)),
        "transition_max_diff": float(np.max(differences)),
    }


def contact_sheet(frames_rgb: Sequence[np.ndarray], *, tile_width: int = 256) -> Image.Image:
    tiles = [
        Image.fromarray(center_crop_resize(frame, width=tile_width)).convert("RGB")
        for frame in frames_rgb
    ]
    tile_height = tiles[0].height
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 2), "black")
    for index, tile in enumerate(tiles[:6]):
        canvas.paste(ImageOps.fit(tile, (tile_width, tile_height)), ((index % 3) * tile_width, (index // 3) * tile_height))
    return canvas


def load_video_sample(
    row: dict[str, str],
    *,
    manifest: Path,
    sample_frames: int,
    default_num_frames: int,
    default_fps: float,
) -> tuple[Image.Image | None, dict[str, float], str]:
    try:
        path = resolve_video_path(row, manifest)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or default_fps)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        clip_start = safe_float(row.get("clip_start_sec"), 0.0)
        clip_frames = safe_int(row.get("clip_num_frames"), default_num_frames)
        clip_fps = safe_float(row.get("clip_fps"), default_fps)
        duration = max((clip_frames - 1) / max(clip_fps, 1e-6), 0.0)
        times = np.linspace(clip_start, clip_start + duration, sample_frames)
        indices = np.rint(times * source_fps).astype(np.int64)
        if total_frames > 0:
            indices = np.clip(indices, 0, total_frames - 1)
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"failed reading frame {index} from {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        capture.release()
        metrics = motion_metrics(frames)
        metrics["decoded_frame_count"] = float(len(frames))
        return contact_sheet(frames), metrics, ""
    except Exception as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"


def load_processed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("_score_record_id", ""))
            for row in csv.DictReader(handle)
            if row.get("_score_record_id")
        }


def score_batch(
    rows: Sequence[tuple[int, dict[str, str]]],
    *,
    manifest: Path,
    scorer: SiglipPairScorer,
    executor: ThreadPoolExecutor,
    sample_frames: int,
    default_num_frames: int,
    default_fps: float,
    model_id: str,
) -> list[dict[str, object]]:
    loaded = list(
        executor.map(
            lambda value: load_video_sample(
                value[1],
                manifest=manifest,
                sample_frames=sample_frames,
                default_num_frames=default_num_frames,
                default_fps=default_fps,
            ),
            rows,
        )
    )
    valid = [index for index, (sheet, _, _) in enumerate(loaded) if sheet is not None]
    scored: dict[int, tuple[float, float]] = {}
    if valid:
        images = [loaded[index][0] for index in valid]
        captions = [first_value(rows[index][1], TEXT_COLUMNS) for index in valid]
        scores, logits = scorer.score(images, captions)
        scored = {index: (score, logit) for index, score, logit in zip(valid, scores, logits)}
    output: list[dict[str, object]] = []
    for index, (row_number, row) in enumerate(rows):
        value: dict[str, object] = dict(row)
        value["_score_record_id"] = record_id(row, row_number)
        value["video_path"] = resolve_optional_path(
            first_value(row, ("video_path", "media_path", "video", "path", "mp4")),
            manifest,
        )
        if first_value(row, ("latent_path", "latents")):
            value["latent_path"] = resolve_optional_path(
                first_value(row, ("latent_path", "latents")), manifest
            )
        value.update({field: "" for field in SCORE_FIELDS})
        value["video_score_status"] = "error"
        value["video_score_error"] = loaded[index][2]
        value["siglip_model"] = model_id
        value["siglip_revision"] = scorer.resolved_revision
        if index in scored:
            score, logit = scored[index]
            value.update(loaded[index][1])
            value["siglip_score"] = f"{score:.8f}"
            value["siglip_logit"] = f"{logit:.8f}"
            value["video_score_status"] = "ok"
            value["video_score_error"] = ""
        output.append(value)
    return output


def merge_shards(output_dir: Path, fieldnames: Sequence[str], world_size: int) -> Path:
    output = output_dir / "video_scores.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank in range(world_size):
            shard = output_dir / "shards" / f"video_scores_rank{rank:03d}.csv"
            if not shard.is_file():
                raise FileNotFoundError(shard)
            with shard.open("r", encoding="utf-8", newline="") as source:
                writer.writerows(csv.DictReader(source))
    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", default="data/mobileov_video_cascade_v1")
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch16-384")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--sample-frames", type=int, default=6)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    try:
        manifest = Path(args.input_manifest)
        output_dir = Path(args.output_dir)
        shards_dir = output_dir / "shards"
        if ctx.is_main:
            shards_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Manifest has no header: {manifest}")
            input_fields = list(reader.fieldnames)
            rows = [
                (index, row)
                for index, row in enumerate(reader, start=1)
                if (index - 1) % ctx.world_size == ctx.rank
                and (args.max_samples <= 0 or index <= args.max_samples)
            ]
        output_fields = list(dict.fromkeys([*input_fields, "_score_record_id", *SCORE_FIELDS]))
        shard = shards_dir / f"video_scores_rank{ctx.rank:03d}.csv"
        processed = load_processed(shard) if args.resume else set()
        pending = [value for value in rows if record_id(value[1], value[0]) not in processed]
        mode = "a" if args.resume and shard.is_file() else "w"
        scorer = SiglipPairScorer(
            model_id=args.model_id,
            revision=args.model_revision,
            device=ctx.device,
            dtype=torch_dtype(args.dtype),
        )
        completed = errors = 0
        with ThreadPoolExecutor(max_workers=args.loader_workers) as executor, shard.open(
            mode, encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            for start in range(0, len(pending), args.batch_size):
                results = score_batch(
                    pending[start : start + args.batch_size],
                    manifest=manifest,
                    scorer=scorer,
                    executor=executor,
                    sample_frames=args.sample_frames,
                    default_num_frames=args.num_frames,
                    default_fps=args.fps,
                    model_id=args.model_id,
                )
                writer.writerows(results)
                completed += len(results)
                errors += sum(row["video_score_status"] != "ok" for row in results)
                if completed % args.flush_every < len(results):
                    handle.flush()
                if args.log_every > 0 and completed % args.log_every < len(results):
                    print(
                        f"rank={ctx.rank} scored={completed}/{len(pending)} errors={errors}",
                        flush=True,
                    )
        barrier()
        if ctx.is_main:
            merged = merge_shards(output_dir, output_fields, ctx.world_size)
            metadata = {
                "input_manifest": str(manifest),
                "output_manifest": str(merged),
                "max_samples": args.max_samples,
                "world_size": ctx.world_size,
                "sample_frames": args.sample_frames,
                "model_id": args.model_id,
                "model_revision": scorer.resolved_revision,
                "motion_metrics": (
                    "six-frame RGB difference and Farneback flow; used for filtering, not evaluation"
                ),
            }
            (output_dir / "score_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(metadata, indent=2), flush=True)
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
