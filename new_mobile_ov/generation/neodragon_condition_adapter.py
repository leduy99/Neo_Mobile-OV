from __future__ import annotations

import torch
import torch.nn as nn


class BridgeToNeodragonConditionAdapter(nn.Module):
    """Map Mobile-OV bridge tokens to Neodragon DiT text conditions.

    Mobile-OV bridge tokens are SANA-shaped:

    - tokens: [B, 300, 2304]
    - mask:   [B, 1, 1, 300] or [B, 300]

    Neodragon DiT conditions are:

    - encoder_hidden_states: [B, 128, 1536] after Neodragon's context adapter
    - encoder_attention_mask: [B, 128]
    - pooled_projections: [B, 2048]

    This adapter is intentionally small and readable. It can be trained by
    distilling frozen Neodragon text conditions from captions.
    """

    def __init__(
        self,
        *,
        bridge_dim: int = 2304,
        neodragon_dim: int = 1536,
        pooled_dim: int = 2048,
        num_queries: int = 128,
        num_heads: int = 12,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.neodragon_dim = int(neodragon_dim)
        self.input_norm = nn.LayerNorm(bridge_dim)
        self.kv_proj = nn.Linear(bridge_dim, neodragon_dim, bias=False)
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, neodragon_dim) * 0.02)
        self.query_norm = nn.LayerNorm(neodragon_dim)
        self.kv_norm = nn.LayerNorm(neodragon_dim)
        self.attn = nn.MultiheadAttention(neodragon_dim, num_heads, batch_first=True)
        hidden = int(neodragon_dim * mlp_ratio)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(neodragon_dim),
            nn.Linear(neodragon_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, neodragon_dim),
        )
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(bridge_dim),
            nn.Linear(bridge_dim, pooled_dim),
        )

    @staticmethod
    def normalize_mask(mask: torch.Tensor | None, *, batch: int, length: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, length, dtype=torch.bool, device=device)
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        elif mask.dim() == 3:
            mask = mask.squeeze(1)
        return mask.to(device=device).bool()

    def masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def forward(
        self,
        bridge_tokens: torch.Tensor,
        bridge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bridge_tokens.dim() != 3:
            raise ValueError(f"bridge_tokens must be [B,N,C], got {tuple(bridge_tokens.shape)}")
        batch, length, _ = bridge_tokens.shape
        mask = self.normalize_mask(bridge_mask, batch=batch, length=length, device=bridge_tokens.device)
        kv = self.kv_proj(self.input_norm(bridge_tokens))
        queries = self.queries.expand(batch, -1, -1)
        out, _ = self.attn(
            self.query_norm(queries),
            self.kv_norm(kv),
            self.kv_norm(kv),
            key_padding_mask=~mask,
            need_weights=False,
        )
        out = out + self.token_mlp(out)
        pooled = self.pooled_head(self.masked_mean(bridge_tokens, mask))
        # Neodragon uses a fixed max-length T5 mask. Adapter output always fills it.
        out_mask = torch.ones(batch, self.num_queries, dtype=torch.long, device=bridge_tokens.device)
        return out, out_mask, pooled
