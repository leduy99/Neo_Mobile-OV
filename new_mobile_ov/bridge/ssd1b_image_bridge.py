from __future__ import annotations

import math
from typing import List, NamedTuple

import torch
import torch.nn as nn

from new_mobile_ov.bridge.sana_prompt_bridge import SimpleTokenizer
from new_mobile_ov.checkpoints import ensure_smolvlm2_checkpoint
from new_mobile_ov.config import BridgeConfig, ImageBridgeConfig
from new_mobile_ov.smolvlm2 import SmolVLMModel, load_smolvlm2_from_ckpt


class SSD1BImageCondition(NamedTuple):
    """Condition tensors consumed by SSD1B's SDXL UNet."""

    clip_l_tokens: torch.Tensor
    clip_big_g_tokens: torch.Tensor
    pooled: torch.Tensor

    @property
    def prompt_embeds(self) -> torch.Tensor:
        return torch.cat([self.clip_l_tokens, self.clip_big_g_tokens], dim=-1)


def _detect_hidden_size(model: nn.Module) -> int:
    wrapped = getattr(model, "_model", None)
    wrapped_config = getattr(wrapped, "config", None)
    if wrapped_config is not None:
        hidden_size = getattr(wrapped_config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Could not detect SmolVLM2 text hidden size.")
    return int(hidden_size)


class SmolVLM2TextFeatureProvider(nn.Module):
    """Frozen SmolVLM2 text forward shared by Mobile-OV conditioning heads."""

    def __init__(
        self,
        cfg: BridgeConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.max_length = int(cfg.max_length)
        self.tokenizer_model_id = str(cfg.tokenizer_model_id)
        checkpoint = ensure_smolvlm2_checkpoint(cfg.smolvlm2_ckpt_path)
        self.smolvlm2_model = load_smolvlm2_from_ckpt(
            checkpoint,
            device=device,
            model_class=SmolVLMModel,
        )
        self.smolvlm2_model.eval().requires_grad_(False)
        self.hidden_size = _detect_hidden_size(self.smolvlm2_model)
        self._cached_tokenizer = None
        self.to(device=device, dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.smolvlm2_model.eval()
        return self

    def _tokenizer(self):
        tokenizer = self.smolvlm2_model.get_tokenizer()
        if tokenizer is None:
            if self._cached_tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    self._cached_tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_model_id,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    config = getattr(getattr(self.smolvlm2_model, "_model", None), "config", None)
                    vocab_size = int(getattr(config, "vocab_size", 32000))
                    self._cached_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
            tokenizer = self._cached_tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        return tokenizer

    def forward(self, prompts: List[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        tokenizer = self._tokenizer()
        prompts = [
            str(prompt).strip() if str(prompt).strip() else (tokenizer.eos_token or tokenizer.pad_token or " ")
            for prompt in prompts
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self.smolvlm2_model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self.smolvlm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (list, tuple)):
            raise RuntimeError("SmolVLM2 did not return hidden states.")
        layers = [value for value in hidden_states if isinstance(value, torch.Tensor) and value.dim() == 3]
        if not layers:
            raise RuntimeError("SmolVLM2 returned no token hidden-state tensors.")
        return layers, attention_mask


class ImageQueryBlock(nn.Module):
    """Mask-aware source attention followed by query interaction and an FFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cross, _ = self.cross_attention(
            self.query_norm(queries),
            self.source_norm(source),
            self.source_norm(source),
            key_padding_mask=~source_mask.bool(),
            need_weights=False,
        )
        queries = queries + cross
        self_value, _ = self.self_attention(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            key_padding_mask=None if query_mask is None else ~query_mask.bool(),
            need_weights=False,
        )
        queries = queries + self_value
        queries = queries + self.ff(self.ff_norm(queries))
        if query_mask is not None:
            queries = queries * query_mask.to(dtype=queries.dtype).unsqueeze(-1)
        return queries


class MobileOVSSD1BImageBridge(nn.Module):
    """Map one frozen SmolVLM2 text forward to SSD1B's native dual-CLIP contract."""

    def __init__(
        self,
        bridge_cfg: BridgeConfig,
        image_cfg: ImageBridgeConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge_cfg = bridge_cfg
        self.image_cfg = image_cfg
        self.sequence_length = int(image_cfg.sequence_length)
        self.feature_provider = SmolVLM2TextFeatureProvider(
            bridge_cfg,
            device=device,
            dtype=dtype,
        )
        source_dim = int(self.feature_provider.hidden_size)
        attention_dim = int(image_cfg.attention_dim)
        num_fuse_layers = int(image_cfg.num_fuse_layers)
        if num_fuse_layers < 1:
            raise ValueError("image_bridge.num_fuse_layers must be positive.")

        self.num_fuse_layers = num_fuse_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_fuse_layers))
        self.semantic_norm = nn.LayerNorm(source_dim)
        self.semantic_projection = nn.Linear(source_dim, attention_dim, bias=False)
        self.lexical_norm = nn.LayerNorm(source_dim)
        self.lexical_projection = nn.Linear(source_dim, attention_dim, bias=False)
        gate = min(max(float(image_cfg.lexical_gate_init), 1e-4), 1.0 - 1e-4)
        self.lexical_gate_logit = nn.Parameter(
            torch.tensor([math.log(gate / (1.0 - gate))], dtype=torch.float32)
        )
        self.source_output_norm = nn.LayerNorm(attention_dim)

        # The final query is global-only; the first 77 queries reproduce CLIP token slots.
        self.queries = nn.Parameter(
            torch.randn(1, self.sequence_length + 1, attention_dim, dtype=torch.float32) * 0.02
        )
        self.query_blocks = nn.ModuleList(
            [
                ImageQueryBlock(
                    attention_dim,
                    num_heads=int(image_cfg.num_heads),
                    ff_mult=int(image_cfg.ff_mult),
                    dropout=float(image_cfg.dropout),
                )
                for _ in range(int(image_cfg.num_layers))
            ]
        )
        self.clip_l_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, int(image_cfg.clip_l_dim)),
        )
        self.clip_big_g_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, int(image_cfg.clip_big_g_dim)),
        )
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, int(image_cfg.pooled_dim)),
        )
        self.to(device=device, dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_provider.smolvlm2_model.eval()
        return self

    def promote_trainable_parameters_to_fp32(self) -> None:
        """Keep small bridge updates in FP32 while frozen towers remain BF16."""
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        frozen_prefix = "feature_provider.smolvlm2_model."
        return {
            name: value
            for name, value in state.items()
            if not name.startswith(frozen_prefix)
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("feature_provider.smolvlm2_model.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Invalid SSD1B Image Bridge state: missing={missing}, unexpected={unexpected}"
            )

    def _fuse_source(self, hidden_layers: list[torch.Tensor]) -> torch.Tensor:
        if len(hidden_layers) < self.num_fuse_layers:
            raise RuntimeError(
                f"Need {self.num_fuse_layers} SmolVLM2 layers, got {len(hidden_layers)}."
            )
        weights = torch.softmax(self.layer_weights, dim=0)
        semantic = torch.zeros_like(hidden_layers[-1])
        for weight, hidden in zip(weights, hidden_layers[-self.num_fuse_layers :]):
            semantic = semantic + weight.to(dtype=hidden.dtype) * hidden
        semantic = self.semantic_projection(self.semantic_norm(semantic))
        lexical = self.lexical_projection(self.lexical_norm(hidden_layers[0]))
        lexical_gate = torch.sigmoid(self.lexical_gate_logit).to(dtype=semantic.dtype)
        return self.source_output_norm(semantic + lexical_gate * lexical)

    def forward(self, prompts: List[str]) -> SSD1BImageCondition:
        hidden_layers, source_mask = self.feature_provider(prompts)
        source = self._fuse_source(hidden_layers)
        queries = self.queries.expand(source.shape[0], -1, -1).to(dtype=source.dtype)
        for block in self.query_blocks:
            queries = block(queries, source, source_mask)
        token_queries = queries[:, : self.sequence_length]
        global_query = queries[:, self.sequence_length]
        return SSD1BImageCondition(
            clip_l_tokens=self.clip_l_head(token_queries),
            clip_big_g_tokens=self.clip_big_g_head(token_queries),
            pooled=self.pooled_head(global_query),
        )
