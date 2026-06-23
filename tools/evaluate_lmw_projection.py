#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "sana"))

from diffusion.model.wan.vae import WanVAE
from new_mobile_ov.bridge.text_bridge import MobileOVTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.motion import LatentMotionWeaver
from new_mobile_ov.training.latent_dataset import WanVAELatentDataset
from new_mobile_ov.training.losses import latent_motion_weaver_loss


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "indices": torch.tensor([int(item["index"]) for item in batch], dtype=torch.long),
        "prompts": [str(item["prompt"]) for item in batch],
        "latents": torch.stack([item["latent"] for item in batch], dim=0),
    }


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def save_video(frames_cthw: torch.Tensor, path: Path, fps: int = 8) -> None:
    frames = ((frames_cthw.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
    frames = frames.permute(1, 2, 3, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def decode_latent(vae: WanVAE, z: torch.Tensor) -> torch.Tensor:
    return vae.decode([z.detach().float().to(vae.device)])[0]


def get_text(
    text_mode: str,
    batch: dict[str, object],
    *,
    text_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    stable_text: torch.Tensor | None,
    bridge: MobileOVTextBridge | None,
) -> torch.Tensor:
    z_gt = batch["latents"]
    if text_mode == "random":
        return torch.randn(z_gt.shape[0], text_dim, device=device, dtype=dtype)
    if text_mode == "zero":
        return torch.zeros(z_gt.shape[0], text_dim, device=device, dtype=dtype)
    if text_mode == "stable_random":
        assert stable_text is not None
        return stable_text[batch["indices"]].to(device=device, dtype=dtype)
    assert bridge is not None
    with torch.no_grad():
        _, _, text = bridge.encode(batch["prompts"])
    return text


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not items:
        return out
    keys = sorted(items[0].keys())
    for key in keys:
        values = np.asarray([item[key] for item in items], dtype=np.float64)
        out[key] = float(values.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_current.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-mode", choices=["checkpoint", "bridge", "zero", "random", "stable_random"], default="checkpoint")
    parser.add_argument("--text-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-path", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth")
    parser.add_argument("--decode-count", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", cfg)
    if args.manifest:
        cfg.data.latent_manifest = args.manifest
    if args.output_dir:
        cfg.data.output_dir = args.output_dir

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_name(cfg.train.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    out_dir = Path(cfg.data.output_dir)
    decode_dir = out_dir / "decoded_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_mode = str(ckpt.get("text_mode", "zero")) if args.text_mode == "checkpoint" else args.text_mode
    text_seed = int(ckpt.get("text_seed", 0)) if args.text_seed is None else int(args.text_seed)

    dataset = WanVAELatentDataset(cfg.data.latent_manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    bridge = MobileOVTextBridge(cfg.bridge, device=device, dtype=dtype).eval() if text_mode == "bridge" else None
    stable_text = None
    if text_mode == "stable_random":
        gen = torch.Generator(device="cpu").manual_seed(text_seed)
        stable_text = torch.randn(len(dataset), cfg.motion.text_dim, generator=gen)

    model = LatentMotionWeaver(
        latent_channels=cfg.motion.latent_channels,
        text_dim=cfg.motion.text_dim,
        hidden_dim=cfg.motion.hidden_dim,
        temporal_len=cfg.motion.temporal_len,
        num_blocks=cfg.motion.num_blocks,
        mlp_ratio=cfg.motion.mlp_ratio,
        temporal_kernel=cfg.motion.temporal_kernel,
        temporal_dilation_cycle=cfg.motion.temporal_dilation_cycle,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    vae = WanVAE(vae_pth=args.vae_path, dtype=dtype, device=device) if args.decode_count > 0 else None

    rows = []
    copy_items: list[dict[str, float]] = []
    pred_items: list[dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluate LMW")):
            z_gt = batch["latents"].to(device=device, dtype=dtype)
            text = get_text(
                text_mode,
                batch,
                text_dim=cfg.motion.text_dim,
                device=device,
                dtype=dtype,
                stable_text=stable_text,
                bridge=bridge,
            )
            z_copy = z_gt[:, :, 0:1].repeat(1, 1, z_gt.shape[2], 1, 1)
            z_pred = model(z_gt[:, :, 0], text)

            copy_loss, copy_metrics = latent_motion_weaver_loss(z_copy.float(), z_gt.float(), cfg.train.motion_weight, cfg.train.accel_weight)
            pred_loss, pred_metrics = latent_motion_weaver_loss(z_pred.float(), z_gt.float(), cfg.train.motion_weight, cfg.train.accel_weight)
            copy_item = {
                "loss": float(copy_loss.cpu()),
                "latent_loss": float(copy_metrics["latent_loss"].cpu()),
                "motion_loss": float(copy_metrics["motion_loss"].cpu()),
            }
            pred_item = {
                "loss": float(pred_loss.cpu()),
                "latent_loss": float(pred_metrics["latent_loss"].cpu()),
                "motion_loss": float(pred_metrics["motion_loss"].cpu()),
            }
            copy_items.append(copy_item)
            pred_items.append(pred_item)
            rows.append(
                {
                    "index": int(batch["indices"][0]),
                    "copy_loss": copy_item["loss"],
                    "pred_loss": pred_item["loss"],
                    "copy_latent_loss": copy_item["latent_loss"],
                    "pred_latent_loss": pred_item["latent_loss"],
                    "copy_motion_loss": copy_item["motion_loss"],
                    "pred_motion_loss": pred_item["motion_loss"],
                }
            )

            if vae is not None and batch_idx < args.decode_count:
                gt_video = decode_latent(vae, z_gt[0])
                copy_video = decode_latent(vae, z_copy[0])
                pred_video = decode_latent(vae, z_pred[0])
                save_video(gt_video, decode_dir / f"{batch_idx:04d}_gt.mp4", fps=args.fps)
                save_video(copy_video, decode_dir / f"{batch_idx:04d}_copy.mp4", fps=args.fps)
                save_video(pred_video, decode_dir / f"{batch_idx:04d}_pred.mp4", fps=args.fps)

    copy_mean = mean_metrics(copy_items)
    pred_mean = mean_metrics(pred_items)
    summary = {
        "num_samples": len(dataset),
        "checkpoint": str(args.checkpoint),
        "manifest": str(cfg.data.latent_manifest),
        "text_mode": text_mode,
        "text_seed": text_seed,
        "copy": copy_mean,
        "pred": pred_mean,
        "improvement": {
            key: float((copy_mean[key] - pred_mean[key]) / max(copy_mean[key], 1e-8))
            for key in copy_mean
            if key in pred_mean
        },
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "eval_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
