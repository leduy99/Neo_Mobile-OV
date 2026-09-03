#!/usr/bin/env python
# ruff: noqa: E402
"""Create deployment-matched DreamLite-anchor/NeoDragon-teacher trajectories."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation.backends import DreamLiteMobileBackend
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    setup_distributed,
)
from tools.prepare_neodragon_dmd_synthetic_data import dtype_from_name, load_teacher


PROMPT_COLUMNS = ("prompt", "caption_long", "caption_medium", "caption", "text")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def first_value(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = normalize_text(row.get(column, ""))
        if value:
            return value
    return ""


def load_prompt_rows(path: Path, max_samples: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prompt bank has no header: {path}")
        rows = []
        for index, row in enumerate(reader):
            prompt = first_value(row, PROMPT_COLUMNS)
            if not prompt:
                continue
            value = {str(key): str(item or "") for key, item in row.items() if key}
            value["prompt"] = prompt
            value.setdefault("prompt_id", hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24])
            value["prompt_bank_row"] = str(index + 1)
            rows.append(value)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable prompts in {path}")
    return rows


def anchor_path(output_dir: Path, index: int, prompt_id: str) -> Path:
    return output_dir / "anchors" / f"anchor_{index:07d}_{prompt_id[:12]}.png"


def trajectory_path(output_dir: Path, index: int) -> Path:
    return output_dir / "latents" / f"sample_{index:07d}.pt"


def valid_anchor(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size
    except Exception:
        return False


def valid_trajectory(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload.get("latent") if isinstance(payload, dict) else None
        return (
            isinstance(latent, torch.Tensor)
            and latent.ndim == 4
            and latent.shape[1] == 7
            and payload.get("pair_contract") == "same_dreamlite_anchor_native_teacher"
        )
    except Exception:
        return False


def atomic_save(payload: object, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_bridge(
    config_path: str,
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    cfg = load_config(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid DreamLite bridge checkpoint: {checkpoint_path}")
    bridge = MobileOVDreamLiteImageBridge(
        cfg.bridge, cfg.dreamlite_bridge, device=device, dtype=dtype
    ).eval()
    bridge.load_trainable_state_dict(state)
    backend = DreamLiteMobileBackend(
        cfg.dreamlite, device=device, dtype=dtype, load_vae=True
    )
    return bridge, backend, payload


@torch.inference_mode()
def generate_anchors(args, rows, ctx, dtype: torch.dtype) -> None:
    expected_size = (args.anchor_width, args.anchor_height)
    pending = [
        index
        for index in range(ctx.rank, len(rows), ctx.world_size)
        if args.overwrite
        or not valid_anchor(
            anchor_path(Path(args.output_dir), index, rows[index]["prompt_id"]),
            expected_size,
        )
    ]
    if not pending:
        rank0_print(ctx, "DreamLite anchor phase already complete for this rank.")
        return
    bridge, backend, payload = load_image_bridge(
        args.image_config,
        Path(args.image_bridge_checkpoint),
        device=ctx.device,
        dtype=dtype,
    )
    checkpoint_step = int(payload.get("step", -1)) if isinstance(payload, dict) else -1
    rank0_print(
        ctx,
        f"DreamLite anchor phase: total={len(rows)} pending_rank0={len(pending)} "
        f"checkpoint_step={checkpoint_step}",
    )
    iterator = tqdm(pending, desc="DreamLite anchors", disable=not ctx.is_main)
    for completed, index in enumerate(iterator, start=1):
        row = rows[index]
        seed = int(args.seed) + index
        with torch.random.fork_rng(devices=[ctx.device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.autocast("cuda", dtype=dtype):
                condition = bridge([row["prompt"]], mode="generate")
                image = backend.generate_images(
                    condition,
                    height=args.anchor_height,
                    width=args.anchor_width,
                    time_id_height=args.anchor_time_id_height,
                    time_id_width=args.anchor_time_id_width,
                    num_steps=args.image_steps,
                    seed=seed,
                )[0].convert("RGB")
        destination = anchor_path(Path(args.output_dir), index, row["prompt_id"])
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.png")
        image.save(temporary, format="PNG", optimize=False)
        temporary.replace(destination)
        if completed % args.log_every == 0:
            iterator.set_postfix(saved=completed)
    del bridge, backend, payload
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_teacher_trajectories(args, rows, ctx, dtype: torch.dtype) -> str:
    pending = [
        index
        for index in range(ctx.rank, len(rows), ctx.world_size)
        if args.overwrite or not valid_trajectory(trajectory_path(Path(args.output_dir), index))
    ]
    teacher = load_teacher(load_config(args.video_config), device=ctx.device, dtype=dtype)
    rank0_print(
        ctx,
        f"Native teacher phase: total={len(rows)} pending_rank0={len(pending)} "
        "contract=same DreamLite anchor -> native monolithic continuation",
    )
    iterator = tqdm(pending, desc="Matched teacher trajectories", disable=not ctx.is_main)
    for completed, index in enumerate(iterator, start=1):
        row = rows[index]
        prompt = row["prompt"]
        seed = int(args.seed) + index
        source_anchor = anchor_path(Path(args.output_dir), index, row["prompt_id"])
        if not valid_anchor(source_anchor, (args.anchor_width, args.anchor_height)):
            raise RuntimeError(f"Missing or invalid source anchor: {source_anchor}")
        with Image.open(source_anchor) as image_handle:
            image = image_handle.convert("RGB").copy()
        with torch.random.fork_rng(devices=[ctx.device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            latent = teacher["generate"](
                teacher["text"],
                teacher["dit"],
                teacher["adapter"],
                teacher["vae"],
                teacher["scheduler"],
                prompt=prompt,
                image=image,
                height=args.video_height,
                width=args.video_width,
                num_frames=args.num_frames,
                num_inference_steps=[args.first_unit_steps] * 3,
                video_num_inference_steps=[args.video_unit_steps] * 3,
                do_classifier_free_guidance=True,
                guidance_scale=args.guidance_scale,
                video_guidance_scale=args.video_guidance_scale,
                prompt_modifier=teacher["prompt_modifier"],
                negative_prompt=teacher["negative_prompt"],
                output_type="latent",
                device=ctx.device,
                dtype=dtype,
            )
        latent = latent.detach().cpu().contiguous()
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 7:
            raise RuntimeError(
                f"Expected matched teacher latent [1,C,7,H,W], got {tuple(latent.shape)}"
            )
        atomic_save(
            {
                "latent": latent[0],
                "index": index,
                "prompt_id": row["prompt_id"],
                "prompt": prompt,
                "condition_prompt": prompt + str(teacher["prompt_modifier"]),
                "anchor_image_path": str(source_anchor.relative_to(Path(args.output_dir))),
                "anchor_sha256": image_sha256(source_anchor),
                "pair_contract": "same_dreamlite_anchor_native_teacher",
                "image_generator": "dreamlite_mobile_v11_balanced_bridge",
                "teacher": "released_neodragon_monolithic_multistep_cfg",
                "seed": seed,
            },
            trajectory_path(Path(args.output_dir), index),
        )
        if completed % args.log_every == 0:
            iterator.set_postfix(saved=completed)
    model_path = str(teacher["model_path"])
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return model_path


def write_manifest(args, rows, *, native_model_path: str | None) -> int:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.csv"
    fields = (
        "sample_id",
        "prompt_id",
        "latent_path",
        "latent_shape",
        "prompt",
        "caption_short",
        "caption_medium",
        "caption_long",
        "condition_prompt",
        "anchor_image_path",
        "anchor_sha256",
        "source_name",
        "source_key",
        "source_manifest",
        "source_row",
        "capabilities",
        "seed",
        "teacher",
        "image_generator",
        "image_bridge_checkpoint",
        "width",
        "height",
        "num_frames",
        "fps",
        "verification_status",
        "verification_source",
    )
    temporary = manifest_path.with_suffix(".csv.tmp")
    ready = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            path = trajectory_path(output_dir, index)
            if not valid_trajectory(path):
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            latent = payload["latent"]
            writer.writerow(
                {
                    "sample_id": index,
                    "prompt_id": row["prompt_id"],
                    "latent_path": str(path.relative_to(output_dir)),
                    "latent_shape": "x".join(str(int(value)) for value in latent.shape),
                    "prompt": row["prompt"],
                    "caption_short": row.get("caption_short", row["prompt"]),
                    "caption_medium": row.get("caption_medium", row["prompt"]),
                    "caption_long": row.get("caption_long", row["prompt"]),
                    "condition_prompt": payload["condition_prompt"],
                    "anchor_image_path": payload["anchor_image_path"],
                    "anchor_sha256": payload["anchor_sha256"],
                    "source_name": row.get("source_name", ""),
                    "source_key": row.get("source_key", ""),
                    "source_manifest": row.get("source_manifest", str(args.prompts)),
                    "source_row": row.get("source_row", row["prompt_bank_row"]),
                    "capabilities": row.get("capabilities", ""),
                    "seed": payload["seed"],
                    "teacher": payload["teacher"],
                    "image_generator": payload["image_generator"],
                    "image_bridge_checkpoint": str(Path(args.image_bridge_checkpoint)),
                    "width": args.video_width,
                    "height": args.video_height,
                    "num_frames": args.num_frames,
                    "fps": args.fps,
                    "verification_status": "pair_contract_verified",
                    "verification_source": "machine",
                }
            )
            ready += 1
    temporary.replace(manifest_path)
    metadata = {
        "dataset": "MobileOV-Data-v1",
        "pool": "video_anchor_teacher",
        "status": "complete" if ready == len(rows) else "partial",
        "pair_contract": "same_dreamlite_anchor_native_teacher",
        "prompt_bank": str(Path(args.prompts)),
        "requested": len(rows),
        "ready": ready,
        "image_config": args.image_config,
        "image_bridge_checkpoint": str(Path(args.image_bridge_checkpoint)),
        "image_bridge_checkpoint_sha256": image_sha256(Path(args.image_bridge_checkpoint)),
        "native_teacher_model_path": native_model_path,
        "video_config": args.video_config,
        "anchor_resolution": [args.anchor_width, args.anchor_height],
        "video_resolution": [args.video_width, args.video_height],
        "num_frames": args.num_frames,
        "fps": args.fps,
        "teacher_steps": {
            "first_unit": args.first_unit_steps,
            "video_unit": args.video_unit_steps,
        },
        "cfg": {
            "guidance_scale": args.guidance_scale,
            "video_guidance_scale": args.video_guidance_scale,
        },
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", default="data/mobileov_anchor_teacher_v1")
    parser.add_argument("--image-config", default="configs/mobile_ov_dreamlite_compact_v9.yaml")
    parser.add_argument("--image-bridge-checkpoint", required=True)
    parser.add_argument("--video-config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--anchor-width", type=int, default=1024)
    parser.add_argument("--anchor-height", type=int, default=640)
    parser.add_argument("--anchor-time-id-width", type=int, default=1280)
    parser.add_argument("--anchor-time-id-height", type=int, default=800)
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--first-unit-steps", type=int, default=20)
    parser.add_argument("--video-unit-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--video-guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--phase", choices=("all", "anchors", "teacher", "finalize"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    try:
        output_dir = Path(args.output_dir)
        if ctx.is_main:
            (output_dir / "anchors").mkdir(parents=True, exist_ok=True)
            (output_dir / "latents").mkdir(parents=True, exist_ok=True)
        barrier()
        rows = load_prompt_rows(Path(args.prompts), args.max_samples)
        dtype = dtype_from_name(args.dtype)
        native_model_path = None
        if args.phase in {"all", "anchors"}:
            generate_anchors(args, rows, ctx, dtype)
            barrier()
        if args.phase in {"all", "teacher"}:
            native_model_path = generate_teacher_trajectories(args, rows, ctx, dtype)
            barrier()
        if ctx.is_main and args.phase in {"all", "teacher", "finalize"}:
            ready = write_manifest(args, rows, native_model_path=native_model_path)
            rank0_print(ctx, f"Finalized matched trajectories: {ready}/{len(rows)}")
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
