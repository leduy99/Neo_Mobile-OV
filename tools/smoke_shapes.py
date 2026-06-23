#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_mobile_ov.config import load_config
from new_mobile_ov.motion import LatentMotionWeaver
from new_mobile_ov.training.losses import latent_motion_weaver_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_current.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = LatentMotionWeaver(
        latent_channels=cfg.motion.latent_channels,
        text_dim=cfg.motion.text_dim,
        hidden_dim=cfg.motion.hidden_dim,
        temporal_len=cfg.motion.temporal_len,
        num_blocks=2,
        mlp_ratio=cfg.motion.mlp_ratio,
        temporal_kernel=cfg.motion.temporal_kernel,
        temporal_dilation_cycle=cfg.motion.temporal_dilation_cycle,
    ).to(device)
    z_gt = torch.randn(2, cfg.motion.latent_channels, cfg.motion.temporal_len, 12, 16, device=device)
    text = torch.randn(2, cfg.motion.text_dim, device=device)
    z_pred = model(z_gt[:, :, 0], text)
    loss, metrics = latent_motion_weaver_loss(z_pred, z_gt)
    print("z_pred", tuple(z_pred.shape))
    print("loss", float(loss.detach().cpu()))
    print({k: float(v.cpu()) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
