#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import barrier, cleanup_distributed, setup_distributed
from tools.train_neodragon_dit_bridge import dtype_from_name, read_video_clip, scale_vae_latents


FAILED_COLUMNS = ["sample_id", "video_path", "error"]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def _valid_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_vae(cfg: Any, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import VAE_ID
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

    vae = AsymmetricCausalVideoVAE.from_pretrained(f"{local_model_path}/{VAE_ID}", torch_dtype=dtype).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def shape_string(tensor: torch.Tensor) -> str:
    return "x".join(str(int(x)) for x in tensor.shape)


def prepare_rows(manifest: Path, max_samples: int) -> list[dict[str, Any]]:
    df = pd.read_csv(manifest)
    if max_samples > 0:
        df = df.head(max_samples)
    if "video_path" not in df.columns:
        raise ValueError(f"{manifest} must contain video_path.")
    if not any(c in df.columns for c in ["prompt", "caption", "text", "caption_long", "caption_medium", "caption_short"]):
        raise ValueError(f"{manifest} must contain a prompt/caption column.")
    return df.to_dict("records")


def output_row_from_source(
    row: dict[str, Any],
    *,
    sample_id: int,
    latent_rel: str,
    latent: torch.Tensor,
    source_manifest: Path,
) -> dict[str, Any]:
    prompt = (
        _valid_text(row.get("prompt"))
        or _valid_text(row.get("caption_long"))
        or _valid_text(row.get("caption_medium"))
        or _valid_text(row.get("caption_short"))
        or _valid_text(row.get("caption"))
        or _valid_text(row.get("text"))
    )
    out = {
        "sample_id": int(sample_id),
        "latent_path": latent_rel,
        "latent_shape": shape_string(latent),
        "prompt": prompt,
        "caption": _valid_text(row.get("caption")) or prompt,
        "video_path": _valid_text(row.get("video_path")),
        "source_manifest": str(source_manifest),
        "clip_start_sec": _safe_float(row.get("clip_start_sec"), 0.0),
        "clip_end_sec": _safe_float(row.get("clip_end_sec"), 0.0),
        "clip_num_frames": _safe_int(row.get("clip_num_frames"), 49),
        "clip_fps": _safe_float(row.get("clip_fps"), 24.0),
    }
    for col in ["caption_short", "caption_medium", "caption_long"]:
        if col in row:
            out[col] = _valid_text(row.get(col))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-encode OpenVid clips into NeoDragon VAE latents.")
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True, help="Prepared OpenVid/NeoDragon clip manifest.")
    parser.add_argument("--output-dir", default="data/openvid_neodragon_2s_latents")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--latent-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ctx = setup_distributed()
    cfg = load_config(args.config)
    dtype = dtype_from_name(args.dtype or cfg.backend.dtype)
    if ctx.device.type == "cpu":
        dtype = torch.float32
    latent_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.latent_dtype]

    manifest = Path(args.manifest).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    latents_dir = out_dir / "latents"
    shards_dir = out_dir / "shards"
    failed_dir = out_dir / "failed"
    if ctx.is_main:
        latents_dir.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    rows = prepare_rows(manifest, args.max_samples)
    assigned = [(idx, row) for idx, row in enumerate(rows) if idx % ctx.world_size == ctx.rank]
    print(
        f"[rank {ctx.rank}/{ctx.world_size}] encoding rows={len(assigned)} total={len(rows)} "
        f"device={ctx.device} dtype={dtype} latent_dtype={latent_dtype}",
        flush=True,
    )
    vae = load_vae(cfg, ctx.device, dtype)

    out_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    iterator = tqdm(assigned, desc=f"rank{ctx.rank} NeoDragon VAE", disable=ctx.rank != 0)
    for idx, row in iterator:
        latent_rel = f"latents/{idx:09d}.pt"
        latent_path = out_dir / latent_rel
        try:
            if latent_path.exists() and latent_path.stat().st_size > 0 and not args.overwrite:
                payload = torch.load(latent_path, map_location="cpu", weights_only=False)
                latent = payload.get("latent", payload) if isinstance(payload, dict) else payload
            else:
                video_path = _valid_text(row.get("video_path"))
                if not video_path:
                    raise RuntimeError("missing video_path")
                frames = read_video_clip(
                    video_path,
                    num_frames=_safe_int(row.get("clip_num_frames"), args.num_frames) or args.num_frames,
                    height=args.height,
                    width=args.width,
                    target_fps=_safe_float(row.get("clip_fps"), args.target_fps) or args.target_fps,
                    clip_start_sec=_safe_float(row.get("clip_start_sec"), 0.0),
                )
                video = frames.unsqueeze(0).to(device=ctx.device, dtype=dtype)
                with torch.no_grad():
                    latent_b = vae.encode(video, temporal_chunk=True).latent_dist.sample()
                    latent_b = scale_vae_latents(latent_b)
                latent = latent_b[0].detach().to(device="cpu", dtype=latent_dtype).contiguous()
                payload = {
                    "latent": latent,
                    "sample_index": int(idx),
                    "source_video_path": video_path,
                    "clip_start_sec": _safe_float(row.get("clip_start_sec"), 0.0),
                    "clip_num_frames": _safe_int(row.get("clip_num_frames"), args.num_frames) or args.num_frames,
                    "clip_fps": _safe_float(row.get("clip_fps"), args.target_fps) or args.target_fps,
                    "height": int(args.height),
                    "width": int(args.width),
                    "scaled_for_neodragon": True,
                }
                tmp = latent_path.with_name(f".{latent_path.name}.rank{ctx.rank}.tmp")
                torch.save(payload, tmp)
                tmp.replace(latent_path)
            out_rows.append(
                output_row_from_source(
                    row,
                    sample_id=idx,
                    latent_rel=latent_rel,
                    latent=latent,
                    source_manifest=manifest,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "sample_id": int(idx),
                    "video_path": _valid_text(row.get("video_path")),
                    "error": repr(exc),
                }
            )

    shard_csv = shards_dir / f"latent_manifest_rank{ctx.rank:05d}.csv"
    failed_csv = failed_dir / f"failed_rank{ctx.rank:05d}.csv"
    pd.DataFrame(out_rows).to_csv(shard_csv, index=False)
    pd.DataFrame(failed, columns=FAILED_COLUMNS).to_csv(failed_csv, index=False)
    barrier()

    if ctx.is_main:
        shard_paths = sorted(shards_dir.glob("latent_manifest_rank*.csv"))
        parts = [read_csv_or_empty(path) for path in shard_paths]
        parts = [df for df in parts if not df.empty]
        merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not merged.empty:
            merged = merged.sort_values("sample_id").reset_index(drop=True)
        latent_manifest = out_dir / "latent_manifest.csv"
        merged.to_csv(latent_manifest, index=False)

        failed_paths = sorted(failed_dir.glob("failed_rank*.csv"))
        failed_parts = [read_csv_or_empty(path) for path in failed_paths]
        failed_parts = [df for df in failed_parts if not df.empty]
        failed_merged = pd.concat(failed_parts, ignore_index=True) if failed_parts else pd.DataFrame()
        failed_out = out_dir / "failed.csv"
        failed_merged.to_csv(failed_out, index=False)

        summary = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(manifest),
            "latent_manifest": str(latent_manifest),
            "output_dir": str(out_dir),
            "rows": int(len(merged)),
            "failed": int(len(failed_merged)),
            "world_size": int(ctx.world_size),
            "latent_dtype": args.latent_dtype,
            "num_frames": int(args.num_frames),
            "height": int(args.height),
            "width": int(args.width),
            "target_fps": float(args.target_fps),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
