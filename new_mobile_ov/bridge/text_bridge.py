from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from new_mobile_ov.config import BridgeConfig
from new_mobile_ov.bridge.sana_prompt_bridge import SanaPromptBridge


def pool_prompt_tokens(prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked mean-pool SANA-style prompt tokens into one conditioning vector."""
    if prompt_mask is None:
        return prompt_embeds.mean(dim=1)
    if prompt_mask.dim() == 4:
        prompt_mask = prompt_mask.squeeze(1).squeeze(1)
    elif prompt_mask.dim() == 3:
        prompt_mask = prompt_mask.squeeze(1)
    mask = prompt_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (prompt_embeds * mask.unsqueeze(-1)).sum(dim=1) / denom


class MobileOVTextBridge(nn.Module):
    """SmolVLM2 + Bridge v2 wrapper used by both generation branches."""

    def __init__(self, cfg: BridgeConfig, device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.bridge = SanaPromptBridge(
            smolvlm2_ckpt_path=cfg.smolvlm2_ckpt_path,
            adapter_ckpt_dir=cfg.adapter_ckpt_dir,
            adapter_in_channels=960,
            adapter_out_channels=cfg.caption_channels,
            adapter_query_length=128,
            force_adapter_query_length=128,
            num_prompt_queries=cfg.sana_model_max_length,
            caption_channels=cfg.caption_channels,
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
            strict_sana_parity_text_path=cfg.strict_sana_parity_text_path,
            strict_sana_use_full_text_window=cfg.strict_sana_use_full_text_window,
            strict_sana_token_select_strategy=cfg.strict_sana_token_select_strategy,
            strict_sana_head_tokens=cfg.strict_sana_head_tokens,
            strict_sana_tail_tokens=cfg.strict_sana_tail_tokens,
            fail_fast_mask=cfg.fail_fast_mask,
            sana_model_max_length=cfg.sana_model_max_length,
        )

    @torch.no_grad()
    def encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeds, prompt_mask = self.bridge(prompts, return_mask=True)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        prompt_mask = prompt_mask.to(device=self.device)
        pooled = pool_prompt_tokens(prompt_embeds, prompt_mask)
        return prompt_embeds, prompt_mask, pooled
