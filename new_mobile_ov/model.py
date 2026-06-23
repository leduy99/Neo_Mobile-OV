from __future__ import annotations

import torch
import torch.nn as nn

from new_mobile_ov.config import NewMobileOVConfig
from new_mobile_ov.bridge import MobileOVTextBridge
from new_mobile_ov.generation import build_generation_backend
from new_mobile_ov.motion import LatentMotionWeaver


class NewMobileOV(nn.Module):
    """Full model container for the new Mobile-OV generation research branch."""

    def __init__(self, cfg: NewMobileOVConfig, device: torch.device | None = None):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge = MobileOVTextBridge(cfg.bridge, device=self.device)
        self.anchor_backend = build_generation_backend(cfg.backend, device=self.device)
        self.motion_weaver = LatentMotionWeaver(
            latent_channels=cfg.motion.latent_channels,
            text_dim=cfg.motion.text_dim,
            hidden_dim=cfg.motion.hidden_dim,
            temporal_len=cfg.motion.temporal_len,
            num_blocks=cfg.motion.num_blocks,
            mlp_ratio=cfg.motion.mlp_ratio,
            temporal_kernel=cfg.motion.temporal_kernel,
            temporal_dilation_cycle=cfg.motion.temporal_dilation_cycle,
        ).to(self.device)

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bridge.encode(prompts)

    def weave_from_gt_anchor(self, z_gt: torch.Tensor, text_pooled: torch.Tensor) -> torch.Tensor:
        z0 = z_gt[:, :, 0]
        return self.motion_weaver(z0, text_pooled)
