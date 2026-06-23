#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_mobile_ov.bridge.text_bridge import MobileOVTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.motion import LatentMotionWeaver
from new_mobile_ov.training.latent_dataset import WanVAELatentDataset
from new_mobile_ov.training.losses import latent_motion_weaver_loss


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
    prompts = [str(item["prompt"]) for item in batch]
    latents = torch.stack([item["latent"] for item in batch], dim=0)
    return {"indices": indices, "prompts": prompts, "latents": latents}


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_current.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-mode", choices=["bridge", "zero", "random", "stable_random"], default="zero")
    parser.add_argument("--text-seed", type=int, default=0)
    parser.add_argument("--random-text", action="store_true", help="Deprecated alias for --text-mode random.")
    parser.add_argument("--manifest", default=None, help="Override data.latent_manifest.")
    parser.add_argument("--output-dir", default=None, help="Override data.output_dir.")
    parser.add_argument("--steps", type=int, default=None, help="Override train.total_steps.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override motion.hidden_dim for smoke tests.")
    parser.add_argument("--blocks", type=int, default=None, help="Override motion.num_blocks for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        cfg.data.latent_manifest = args.manifest
    if args.output_dir:
        cfg.data.output_dir = args.output_dir
    if args.steps is not None:
        cfg.train.total_steps = int(args.steps)
    if args.hidden_dim is not None:
        cfg.motion.hidden_dim = int(args.hidden_dim)
    if args.blocks is not None:
        cfg.motion.num_blocks = int(args.blocks)
    if args.random_text:
        args.text_mode = "random"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_name(cfg.train.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    out_dir = Path(cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = WanVAELatentDataset(cfg.data.latent_manifest)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    batches = itertools.cycle(loader)

    bridge = None
    stable_text = None
    if args.text_mode == "bridge":
        bridge = MobileOVTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    elif args.text_mode == "stable_random":
        gen = torch.Generator(device="cpu").manual_seed(int(args.text_seed))
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=0.0)

    pbar = tqdm(range(1, cfg.train.total_steps + 1), desc="LMW projection")
    for step in pbar:
        batch = next(batches)
        z_gt = batch["latents"].to(device=device, dtype=dtype)
        if z_gt.shape[2] != cfg.motion.temporal_len:
            raise RuntimeError(f"Expected latent T={cfg.motion.temporal_len}, got {z_gt.shape[2]}")
        if args.text_mode == "random":
            text = torch.randn(z_gt.shape[0], cfg.motion.text_dim, device=device, dtype=dtype)
        elif args.text_mode == "zero":
            text = torch.zeros(z_gt.shape[0], cfg.motion.text_dim, device=device, dtype=dtype)
        elif args.text_mode == "stable_random":
            assert stable_text is not None
            text = stable_text[batch["indices"]].to(device=device, dtype=dtype)
        else:
            assert bridge is not None
            with torch.no_grad():
                _, _, text = bridge.encode(batch["prompts"])
        z_pred = model(z_gt[:, :, 0], text)
        loss, metrics = latent_motion_weaver_loss(
            z_pred.float(),
            z_gt.float(),
            motion_weight=cfg.train.motion_weight,
            accel_weight=cfg.train.accel_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.train.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lat=f"{metrics['latent_loss'].item():.4f}", mot=f"{metrics['motion_loss'].item():.4f}")
        if step % cfg.train.save_every == 0 or step == cfg.train.total_steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "config": cfg,
                "text_mode": args.text_mode,
                "text_seed": int(args.text_seed),
            }
            torch.save(ckpt, out_dir / "lmw_latest.pt")
    print(f"Saved latest LMW checkpoint to {out_dir / 'lmw_latest.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
