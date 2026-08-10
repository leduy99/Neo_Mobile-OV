from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from new_mobile_ov.bridge.sana_prompt_bridge import SanaPromptBridge
from new_mobile_ov.bridge.text_bridge import pool_prompt_tokens
from new_mobile_ov.config import BridgeConfig


class MobileOVNeodragonTextBridge(nn.Module):
    """Original SmolVLM2 bridge with tensors matching NeoDragon's condition input.

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(prompts)

    def encode(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, prompt_mask, all_hidden_states = self.encode_smolvlm2_features(prompts)
        return self.encode_from_smolvlm2_features(
            hidden_states,
            prompt_mask,
            all_hidden_states,
        )

    def encode_smolvlm2_features(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor] | None]:
        """Run the frozen SmolVLM2 backbone once using the Exp1 token contract."""

        need_all = self.token_bridge.projector_type in (
            "mcp_tiny",
            "mcp_full",
            "mcp_lexical_gated",
            "mcp_lexical_bottleneck",
        )
        hidden_states, prompt_mask, all_hidden_states = self.token_bridge.encode_prompts(
            prompts,
            return_mask=True,
            return_all_hidden_states=need_all,
        )
        if prompt_mask is None:
            raise RuntimeError("SmolVLM2 feature encoding did not return an attention mask.")
        return hidden_states, prompt_mask, all_hidden_states

    def encode_from_smolvlm2_features(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
        all_hidden_states: List[torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project precomputed Exp1-compatible SmolVLM2 features to NeoDragon."""

        prompt_embeds, prompt_mask = self.token_bridge.project_features(
            hidden_states,
            prompt_mask,
            all_hidden_states,
            return_mask=True,
        )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        prompt_mask = prompt_mask.to(device=self.device)
        pooled_source = pool_prompt_tokens(prompt_embeds, prompt_mask)
        pooled = self.pooled_head(pooled_source)
        return prompt_embeds, prompt_mask, pooled
