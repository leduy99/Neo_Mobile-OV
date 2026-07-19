from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from new_mobile_ov.bridge.sana_prompt_bridge import SanaPromptBridge
from new_mobile_ov.bridge.text_bridge import pool_prompt_tokens
from new_mobile_ov.config import BridgeConfig


class NeoDragonSequenceTranslator(nn.Module):
    """Translate Smol token positions into NeoDragon's 128-token condition sequence."""

    def __init__(self, dim: int, sequence_length: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.query_pos = nn.Parameter(torch.zeros(1, sequence_length, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.normal_(self.query_pos, std=0.02)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.query_pos.shape[1]:
            raise ValueError(
                f"Expected {self.query_pos.shape[1]} source tokens, got {tokens.shape[1]}. "
                "The NeoDragon bridge requires a fixed token sequence."
            )
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match tokens {tuple(tokens.shape[:2])}.")
        queries = tokens + self.query_pos[:, : tokens.shape[1]].to(dtype=tokens.dtype)
        translated, _ = self.cross_attn(
            self.query_norm(queries),
            self.context_norm(tokens),
            self.context_norm(tokens),
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        translated = queries + translated
        return translated + self.ff(self.ff_norm(translated))


class NeoDragonConditionHead(nn.Module):
    """Map normalized MCP features to the raw distribution expected by NeoDragon."""

    def __init__(self, dim: int, bottleneck_dim: int, scale_init: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.residual = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, dim),
        )
        self.output_scale = nn.Parameter(torch.full((dim,), float(scale_init)))
        self.output_bias = nn.Parameter(torch.zeros(dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(tokens)
        residual = self.residual(normalized)
        return normalized * self.output_scale + self.output_bias + residual


class MobileOVNeodragonTextBridge(nn.Module):
    """SmolVLM2 + Bridge v2 with Neodragon DiT-condition-shaped outputs.

    This replaces Neodragon's TextEncoderBundle + ContextAdapter interface.
    The output matches the tensors consumed directly by Neodragon DiT:

    - DiT token condition: [B, 128, 1536]
    - token mask: [B, 128]
    - CLIP-style pooled projection: [B, 2048]

    Keeping this as a bridge variant lets us train from scratch for Neodragon
    without carrying the old SANA [B, 300, 2304] condition shape or Neodragon's
    separate ContextAdapter.
    """

    def __init__(
        self,
        cfg: BridgeConfig,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        *,
        token_dim: int = 1536,
        pooled_dim: int = 2048,
        sequence_length: int = 128,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.token_dim = int(token_dim)
        self.pooled_dim = int(pooled_dim)
        self.sequence_length = int(sequence_length)
        self.use_v2_conditioning = bool(cfg.neodragon_v2_conditioning)

        self.token_bridge = SanaPromptBridge(
            smolvlm2_ckpt_path=cfg.smolvlm2_ckpt_path,
            adapter_ckpt_dir=cfg.adapter_ckpt_dir,
            adapter_in_channels=960,
            adapter_out_channels=self.token_dim,
            adapter_query_length=128,
            force_adapter_query_length=128,
            num_prompt_queries=self.sequence_length,
            caption_channels=self.token_dim,
            precision_dtype=dtype,
            device=self.device,
            tokenizer_model_id=cfg.tokenizer_model_id,
            max_length=cfg.max_length,
            use_vision_head=False,
            projector_type=cfg.projector_type,
            mcp_num_fuse_layers=cfg.mcp_num_fuse_layers,
            mcp_use_refine=cfg.mcp_use_refine,
            mcp_refine_kernel_size=cfg.mcp_refine_kernel_size,
            mcp_lexical_gate_init=cfg.mcp_lexical_gate_init,
            strict_sana_parity_text_path=True,
            strict_sana_use_full_text_window=cfg.strict_sana_use_full_text_window,
            strict_sana_token_select_strategy=cfg.strict_sana_token_select_strategy,
            strict_sana_head_tokens=cfg.strict_sana_head_tokens,
            strict_sana_tail_tokens=cfg.strict_sana_tail_tokens,
            fail_fast_mask=cfg.fail_fast_mask,
            sana_model_max_length=self.sequence_length,
        )
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.pooled_dim),
        )
        if self.use_v2_conditioning:
            self.sequence_translator = NeoDragonSequenceTranslator(
                self.token_dim,
                self.sequence_length,
                int(cfg.neodragon_resampler_heads),
            )
            self.condition_head = NeoDragonConditionHead(
                self.token_dim,
                int(cfg.neodragon_condition_bottleneck_dim),
                float(cfg.neodragon_condition_scale_init),
            )
            self.mask_length_head = nn.Linear(self.token_dim, 1)
            nn.init.zeros_(self.mask_length_head.weight)
            nn.init.zeros_(self.mask_length_head.bias)
        else:
            self.sequence_translator = nn.Identity()
            self.condition_head = nn.Identity()
            self.mask_length_head = None
        self.to(device=self.device, dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        # The backbone is frozen; keeping it in eval mode prevents stochastic
        # hidden states while the bridge heads remain trainable.
        self.token_bridge.smolvlm2_model.eval()
        return self

    def forward(
        self,
        prompts: List[str],
        *,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        return self.encode(prompts, return_aux=return_aux)

    def encode(
        self,
        prompts: List[str],
        *,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        base_tokens, base_mask = self.token_bridge(prompts, return_mask=True)
        base_tokens = base_tokens.to(device=self.device, dtype=self.dtype)
        base_mask = base_mask.to(device=self.device)
        pooled_source = pool_prompt_tokens(base_tokens, base_mask)
        pooled = self.pooled_head(pooled_source)

        aux: dict[str, torch.Tensor] = {"base_mask": base_mask}
        if self.use_v2_conditioning:
            translated = self.sequence_translator(base_tokens, base_mask)
            prompt_embeds = self.condition_head(translated)

            assert self.mask_length_head is not None
            base_lengths = base_mask.float().sum(dim=1)
            max_delta = self.sequence_length * float(self.cfg.neodragon_mask_length_delta_ratio)
            length_delta = torch.tanh(self.mask_length_head(pooled_source).squeeze(-1).float()) * max_delta
            predicted_lengths = (base_lengths + length_delta).clamp(1.0, float(self.sequence_length))
            positions = torch.arange(self.sequence_length, device=base_mask.device, dtype=torch.float32)
            temperature = max(float(self.cfg.neodragon_mask_temperature), 1e-3)
            mask_logits = (predicted_lengths[:, None] - positions[None, :] - 0.5) / temperature
            prompt_mask = (mask_logits >= 0).to(dtype=base_mask.dtype)
            aux.update(
                {
                    "mask_logits": mask_logits,
                    "predicted_lengths": predicted_lengths,
                    "translated_tokens": translated,
                }
            )
        else:
            prompt_embeds = base_tokens
            prompt_mask = base_mask

        result = (prompt_embeds, prompt_mask, pooled)
        if return_aux:
            return (*result, aux)
        return result
