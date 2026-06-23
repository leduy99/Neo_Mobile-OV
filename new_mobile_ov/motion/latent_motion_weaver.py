from __future__ import annotations

import torch
import torch.nn as nn


class MotionWeaveBlock(nn.Module):
    """Conv-heavy latent motion block with text FiLM conditioning."""

    def __init__(
        self,
        dim: int,
        text_dim: int,
        mlp_ratio: int = 4,
        temporal_kernel: int = 3,
        temporal_dilation: int = 1,
    ):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd so the block preserves T.")
        temporal_padding = (temporal_kernel // 2) * int(temporal_dilation)
        self.norm1 = nn.GroupNorm(1, dim)
        self.spatial_dw = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.temporal_dw = nn.Conv3d(
            dim,
            dim,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_padding, 0, 0),
            dilation=(int(temporal_dilation), 1, 1),
            groups=dim,
        )
        self.norm3 = nn.GroupNorm(1, dim)
        self.text_film = nn.Sequential(nn.Linear(text_dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2))
        self.norm4 = nn.GroupNorm(1, dim)
        hidden = dim * mlp_ratio
        self.channel_mlp = nn.Sequential(nn.Conv3d(dim, hidden, 1), nn.SiLU(), nn.Conv3d(hidden, dim, 1))

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial_dw(self.norm1(x))
        x = x + self.temporal_dw(self.norm2(x))
        h = self.norm3(x)
        gamma, beta = self.text_film(text).chunk(2, dim=-1)
        x = x + h * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        return x + self.channel_mlp(self.norm4(x))


class LatentMotionWeaver(nn.Module):
    """Predict a short video latent trajectory from an anchor latent and text condition."""

    def __init__(
        self,
        latent_channels: int = 16,
        text_dim: int = 2304,
        hidden_dim: int = 512,
        temporal_len: int = 21,
        num_blocks: int = 8,
        mlp_ratio: int = 4,
        temporal_kernel: int = 3,
        temporal_dilation_cycle: tuple[int, ...] | list[int] = (1,),
    ):
        super().__init__()
        self.temporal_len = int(temporal_len)
        if not temporal_dilation_cycle:
            temporal_dilation_cycle = (1,)
        self.in_proj = nn.Conv2d(latent_channels, hidden_dim, 1)
        self.temporal_embed = nn.Parameter(torch.zeros(1, hidden_dim, self.temporal_len, 1, 1))
        self.blocks = nn.ModuleList(
            [
                MotionWeaveBlock(
                    hidden_dim,
                    text_dim=text_dim,
                    mlp_ratio=mlp_ratio,
                    temporal_kernel=temporal_kernel,
                    temporal_dilation=int(temporal_dilation_cycle[i % len(temporal_dilation_cycle)]),
                )
                for i in range(num_blocks)
            ]
        )
        self.out_proj = nn.Conv3d(hidden_dim, latent_channels, 1)
        # Start exactly from the copy-anchor baseline; training only learns motion residuals.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        scale = torch.linspace(0.0, 1.0, self.temporal_len)
        self.register_buffer("residual_scale", scale.view(1, 1, self.temporal_len, 1, 1), persistent=False)

    def forward(self, z0: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if z0.dim() != 4:
            raise ValueError(f"z0 must be [B,C,H,W], got {tuple(z0.shape)}")
        x0 = self.in_proj(z0)
        x = x0.unsqueeze(2).repeat(1, 1, self.temporal_len, 1, 1)
        x = x + self.temporal_embed
        for block in self.blocks:
            x = block(x, text)
        delta = self.out_proj(x)
        z0_video = z0.unsqueeze(2).repeat(1, 1, self.temporal_len, 1, 1)
        z_pred = z0_video + self.residual_scale.to(delta.dtype) * delta
        z_pred[:, :, 0] = z0
        return z_pred
