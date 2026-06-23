"""
SANA prompt bridge modules for Q1 distillation.

This module maps SmolVLM2 text embeddings to SANA prompt embeddings.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from new_mobile_ov.bridge.adapter import DM_Adapter
from new_mobile_ov.bridge.smolvlm2_vision_head import SmolVLM2VisionHead
from new_mobile_ov.checkpoints import ensure_smolvlm2_checkpoint
from new_mobile_ov.smolvlm2 import load_smolvlm2_from_ckpt, SmolVLMModel


logger = logging.getLogger(__name__)


def _linspace_unique_positions(length: int, count: int) -> List[int]:
    """Pick `count` approximately-uniform positions from [0, length)."""
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    raw = torch.linspace(0, length - 1, steps=count).round().to(torch.long).tolist()
    out: List[int] = []
    seen: set[int] = set()
    for idx in raw:
        idx = int(max(0, min(length - 1, idx)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < count:
        for idx in range(length):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
                if len(out) >= count:
                    break
    return out[:count]


def build_strict_sana_select_index(
    cur_len: int,
    target_len: int,
    *,
    strategy: str = "tail",
    head_tokens: int = 96,
    tail_tokens: int = 96,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a lightweight token-selection plan for strict SANA parity.

    Strategies:
      - tail: current behavior, keep BOS + tail
      - head_tail: keep BOS + head + tail
      - head_uniform_tail: keep BOS + head + uniformly-sampled middle + tail
    """
    if cur_len < target_len:
        raise RuntimeError(
            f"Strict SANA-parity requires hidden length >= {target_len}, got {cur_len}."
        )
    if cur_len == target_len:
        return torch.arange(cur_len, device=device, dtype=torch.long)

    strategy_key = str(strategy or "tail").strip().lower()
    if strategy_key in {"bos_tail", "tail_only"}:
        strategy_key = "tail"
    if strategy_key not in {"tail", "head_tail", "head_uniform_tail"}:
        raise ValueError(f"Unsupported strict_sana token selection strategy={strategy!r}")

    if strategy_key == "tail":
        tail_len = target_len - 1
        idx = [0] + list(range(cur_len - tail_len, cur_len))
        return torch.tensor(idx, device=device, dtype=torch.long)

    budget = target_len - 1
    head_budget = min(max(int(head_tokens), 0), budget)
    tail_budget = min(max(int(tail_tokens), 0), max(0, budget - head_budget))
    middle_budget = max(0, budget - head_budget - tail_budget)

    head_idx = list(range(1, min(cur_len, 1 + head_budget)))
    tail_start = max(1 + len(head_idx), cur_len - tail_budget)
    tail_idx = list(range(tail_start, cur_len))

    middle_candidates = list(range(1 + len(head_idx), tail_start))
    middle_idx: List[int] = []
    if strategy_key == "head_uniform_tail" and middle_budget > 0 and middle_candidates:
        middle_positions = _linspace_unique_positions(len(middle_candidates), middle_budget)
        middle_idx = [middle_candidates[i] for i in middle_positions]

    selected = [0] + head_idx + middle_idx + tail_idx
    if len(selected) < target_len:
        selected_set = set(selected)
        remaining = [idx for idx in range(1, cur_len) if idx not in selected_set]
        fill_positions = _linspace_unique_positions(len(remaining), target_len - len(selected))
        selected.extend(remaining[i] for i in fill_positions)

    selected = selected[:target_len]
    return torch.tensor(selected, device=device, dtype=torch.long)


class SimpleTokenizer:
    """
    Minimal fallback tokenizer to avoid remote downloads.
    Maps whitespace tokens to ids via hashing within vocab_size.
    """

    def __init__(self, vocab_size: int = 32000, pad_token: str = "<pad>", eos_token: str = "</s>"):
        self.vocab_size = int(max(4, vocab_size))
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _encode(self, text: str, max_length: int | None) -> list[int]:
        tokens = str(text).split()
        ids = []
        for tok in tokens:
            hid = (abs(hash(tok)) % (self.vocab_size - 2)) + 2
            ids.append(hid)
        if max_length is not None:
            ids = ids[: max_length - 1] if max_length > 1 else []
        ids.append(self.eos_token_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        batch_ids = [self._encode(t, max_length if truncation else None) for t in texts]
        max_len = max(len(x) for x in batch_ids) if padding else None
        input_ids = []
        attention = []
        for ids in batch_ids:
            if max_len is not None:
                pad_len = max_len - len(ids)
                input_ids.append(ids + [self.pad_token_id] * pad_len)
                attention.append([1] * len(ids) + [0] * pad_len)
            else:
                input_ids.append(ids)
                attention.append([1] * len(ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SanaBridgeResampler(nn.Module):
    """
    Cross-attention resampler to map adapter tokens -> SANA prompt tokens.
    Input:  [B, K, in_dim]
    Output: [B, Q, out_dim]
    """

    def __init__(
        self,
        in_dim: int = 4096,
        out_dim: int = 2304,
        num_queries: int = 300,
        num_heads: int = 16,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.out_dim = out_dim

        self.queries = nn.Parameter(torch.randn(1, num_queries, out_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.ln_q = nn.LayerNorm(out_dim)
        self.ln_kv = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, int(mlp_mult) * out_dim),
            nn.GELU(),
            nn.Linear(int(mlp_mult) * out_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, K, C] input, got shape {x.shape}")

        batch_size = x.shape[0]
        kv = self.kv_proj(x)
        q = self.queries.expand(batch_size, -1, -1)

        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask.bool()

        out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = out + self.mlp(out)
        return out


class ECALite(nn.Module):
    """Efficient Channel Attention over token-channel sequence [B, N, C]."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=1, keepdim=True)  # [B, 1, C]
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y


class SeqRefine(nn.Module):
    """Depthwise-separable Conv1D refinement along token axis N."""

    def __init__(self, hidden_dim: int, kernel_size: int = 3, use_eca: bool = True):
        super().__init__()
        self.dw = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )
        # Depthwise Conv1d can emit non-contiguous grad strides under DDP bucket view.
        # Force contiguous grad layout to reduce reducer warnings/noise.
        self.dw.weight.register_hook(lambda grad: grad.contiguous())
        # Use token-wise linear instead of 1x1 Conv1D to avoid DDP grad bucket stride mismatch.
        self.pw = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.eca = ECALite(hidden_dim, k_size=3) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)  # [B, C, N]
        y = self.dw(y)
        y = y.transpose(1, 2)
        y = self.pw(y)
        y = torch.nn.functional.hardswish(y)
        y = self.eca(y)
        y = self.norm(y)
        return residual + y


class MCPProjector(nn.Module):
    """
    Mobile Conditioning Projector:
    - Learnable fusion of last K hidden layers
    - Optional lightweight sequence refinement directly in d_vlm space
    - Final projection to d_cond
    """

    def __init__(
        self,
        d_vlm: int,
        d_cond: int,
        d_h: int = 512,
        num_fuse_layers: int = 4,
        use_refine: bool = True,
        refine_kernel_size: int = 3,
        lexical_mode: str = "none",
        lexical_bottleneck_dim: int = 256,
        lexical_gate_init: float = 0.05,
    ):
        super().__init__()
        self.num_fuse_layers = int(max(1, num_fuse_layers))
        self.layer_w = nn.Parameter(torch.zeros(self.num_fuse_layers))
        # No compression / no intermediate expansion:
        # keep features in original VLM width, then project once to conditioning dim.
        self.hidden_dim = int(d_vlm)
        self.mid_dim = 0
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.refine = (
            SeqRefine(self.hidden_dim, kernel_size=refine_kernel_size, use_eca=True)
            if use_refine
            else nn.Identity()
        )
        self.lexical_mode = str(lexical_mode or "none").lower()
        if self.lexical_mode not in {"none", "gated_add", "gated_add_bottleneck"}:
            raise ValueError(f"Unsupported MCP lexical_mode={self.lexical_mode!r}")
        self.lexical_norm = nn.LayerNorm(self.hidden_dim)
        if self.lexical_mode == "gated_add_bottleneck":
            bottleneck_dim = int(max(1, lexical_bottleneck_dim))
            self.lexical_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, bottleneck_dim, bias=False),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, self.hidden_dim, bias=False),
            )
        else:
            self.lexical_proj = nn.Identity()
        lexical_gate_init = float(min(max(lexical_gate_init, 1e-4), 1.0 - 1e-4))
        lexical_gate_logit = math.log(lexical_gate_init / (1.0 - lexical_gate_init))
        self.lexical_gate_logit = nn.Parameter(torch.tensor([lexical_gate_logit], dtype=torch.float32))
        self.out = nn.Linear(self.hidden_dim, d_cond, bias=False)
        self.out_norm = nn.LayerNorm(d_cond)

    def fuse_last_k(self, hidden_layers: List[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
        if len(hidden_layers) < self.num_fuse_layers:
            raise ValueError(
                f"MCP requires at least {self.num_fuse_layers} hidden layers, got {len(hidden_layers)}"
            )
        use_layers = hidden_layers[-self.num_fuse_layers :]
        alpha = torch.softmax(self.layer_w / max(float(temperature), 1e-6), dim=0)
        fused = torch.zeros_like(use_layers[0])
        for a, h in zip(alpha, use_layers):
            fused = fused + a * h
        return fused

    def forward(self, hidden_layers: List[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
        x = self.fuse_last_k(hidden_layers, temperature=temperature)  # [B, N, d_vlm]
        x = self.input_norm(x)
        if self.lexical_mode != "none":
            lexical = self.lexical_norm(hidden_layers[0])
            lexical = self.lexical_proj(lexical)
            lexical_gate = torch.sigmoid(self.lexical_gate_logit).to(device=x.device, dtype=x.dtype)
            x = x + lexical_gate * lexical
        x = torch.nn.functional.hardswish(x)
        x = self.refine(x)
        x = self.out(x)
        x = self.out_norm(x)
        return x


class SanaPromptBridge(nn.Module):
    """
    SmolVLM2 -> VisionHead -> DM_Adapter -> Resampler -> SANA prompt embeddings.
    """

    def __init__(
        self,
        smolvlm2_ckpt_path: str,
        adapter_ckpt_dir: str,
        adapter_in_channels: int = 1152,
        adapter_out_channels: int = 4096,
        adapter_query_length: int = 64,
        adapter_num_encoder_layers: int = 4,
        adapter_num_decoder_layers: int = 4,
        adapter_ff_mult: int = 4,
        smol_vh_num_queries: int = 1,
        num_prompt_queries: int = 300,
        caption_channels: int = 2304,
        precision_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        tokenizer_model_id: str = "HuggingFaceTB/SmolVLM-Instruct",
        force_adapter_query_length: Optional[int] = None,
        max_length: int = 512,
        eps: float = 1e-6,
        use_vision_head: bool = True,
        resampler_num_heads: int = 16,
        resampler_mlp_mult: int = 4,
        lora_enable: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_include_patterns: Optional[List[str]] = None,
        lora_exclude_patterns: Optional[List[str]] = None,
        gate_min_value: float = 0.0,
        projector_type: str = "legacy",
        mcp_hidden_dim: int = 512,
        mcp_num_fuse_layers: int = 2,
        mcp_use_refine: bool = True,
        mcp_refine_kernel_size: int = 3,
        mcp_fusion_temperature: float = 1.0,
        mcp_lexical_bottleneck_dim: int = 256,
        mcp_lexical_gate_init: float = 0.05,
        strict_sana_parity_text_path: bool = False,
        strict_sana_use_full_text_window: bool = False,
        strict_sana_token_select_strategy: str = "tail",
        strict_sana_head_tokens: int = 96,
        strict_sana_tail_tokens: int = 96,
        fail_fast_mask: Optional[bool] = None,
        sana_model_max_length: int = 300,
        sana_chi_prompt: str = "",
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.caption_channels = caption_channels
        self.tokenizer_model_id = tokenizer_model_id
        self.max_length = max_length
        self.gate_min_value = float(gate_min_value)
        self.projector_type = str(projector_type or "legacy").lower()
        self.mcp_fusion_temperature = float(mcp_fusion_temperature)
        self.strict_sana_parity_text_path = bool(strict_sana_parity_text_path)
        self.strict_sana_use_full_text_window = bool(strict_sana_use_full_text_window)
        self.strict_sana_token_select_strategy = str(strict_sana_token_select_strategy or "tail").strip().lower()
        self.strict_sana_head_tokens = int(strict_sana_head_tokens)
        self.strict_sana_tail_tokens = int(strict_sana_tail_tokens)
        self.fail_fast_mask = bool(
            self.strict_sana_parity_text_path if fail_fast_mask is None else fail_fast_mask
        )
        self.sana_model_max_length = int(sana_model_max_length)
        self.sana_chi_prompt = str(sana_chi_prompt or "")
        self._chi_prompt_token_count = None

        smolvlm2_ckpt_path = ensure_smolvlm2_checkpoint(smolvlm2_ckpt_path)
        self.smolvlm2_model = load_smolvlm2_from_ckpt(
            smolvlm2_ckpt_path,
            device=device,
            model_class=SmolVLMModel,
        )
        self.smolvlm2_model.eval().requires_grad_(False)

        smol_hidden_size = None
        if hasattr(self.smolvlm2_model, "_model") and hasattr(self.smolvlm2_model._model, "config"):
            smol_hidden_size = getattr(self.smolvlm2_model._model.config, "hidden_size", None)
        if smol_hidden_size is None:
            cfg = getattr(self.smolvlm2_model, "config", None)
            if cfg is not None and hasattr(cfg, "text_config") and cfg.text_config is not None:
                smol_hidden_size = getattr(cfg.text_config, "hidden_size", None)
            if smol_hidden_size is None:
                smol_hidden_size = getattr(cfg, "hidden_size", 1024)
        if smol_hidden_size != 1024:
            logger.info("Detected SmolVLM2 hidden_size=%s (overriding VisionHead input)", smol_hidden_size)

        self.use_vision_head = bool(use_vision_head)
        if not self.use_vision_head and adapter_in_channels != smol_hidden_size:
            logger.warning(
                "Adapter in_channels (%s) != SmolVLM2 hidden_size (%s); overriding adapter_in_channels.",
                adapter_in_channels,
                smol_hidden_size,
            )
            adapter_in_channels = smol_hidden_size
        if self.use_vision_head:
            self.smolvlm2_vision_head = SmolVLM2VisionHead(
                llm_hidden_size=smol_hidden_size,
                hidden_size=adapter_in_channels,
                learnable_query_length=smol_vh_num_queries,
                TRAINABLE_PRECISION=precision_dtype,
            )
        else:
            self.smolvlm2_vision_head = None

        if self.projector_type in ("mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck"):
            if self.projector_type == "mcp_tiny":
                # Keep tiny projector lightweight but allow K up to 4 for stronger fusion.
                mcp_num_fuse_layers = max(1, min(int(mcp_num_fuse_layers), 4))
                mcp_use_refine = False
            lexical_mode = "none"
            if self.projector_type == "mcp_lexical_gated":
                lexical_mode = "gated_add"
            elif self.projector_type == "mcp_lexical_bottleneck":
                lexical_mode = "gated_add_bottleneck"
            self.projector = MCPProjector(
                d_vlm=smol_hidden_size,
                d_cond=caption_channels,
                d_h=int(mcp_hidden_dim),
                num_fuse_layers=int(mcp_num_fuse_layers),
                use_refine=bool(mcp_use_refine),
                refine_kernel_size=int(mcp_refine_kernel_size),
                lexical_mode=lexical_mode,
                lexical_bottleneck_dim=int(mcp_lexical_bottleneck_dim),
                lexical_gate_init=float(mcp_lexical_gate_init),
            )
            # Keep legacy attrs for backward-compat with training/checkpoint code paths.
            self.adapter = nn.Identity()
            self.adapter_output_norm = nn.Identity()
            self.adapter_output_gate = nn.Parameter(torch.tensor([1.0], dtype=precision_dtype))
            # MCP path does not consume adapter gate in forward; keep it for checkpoint compatibility
            # but freeze to avoid DDP unused-parameter overhead.
            self.adapter_output_gate.requires_grad_(False)
            self.resampler = nn.Identity()
            logger.info(
                (
                    "Using MCP projector: type=%s d_vlm=%s no_compress=True hidden=%s d_cond=%s "
                    "K=%s refine=%s lexical_mode=%s lexical_bottleneck_dim=%s lexical_gate_init=%.4f"
                ),
                self.projector_type,
                smol_hidden_size,
                int(self.projector.hidden_dim),
                caption_channels,
                int(mcp_num_fuse_layers),
                bool(mcp_use_refine),
                lexical_mode,
                int(mcp_lexical_bottleneck_dim),
                float(mcp_lexical_gate_init),
            )
        else:
            if force_adapter_query_length is not None:
                if force_adapter_query_length != adapter_query_length:
                    logger.info(
                        "Adapter query_length forced: %s -> %s",
                        adapter_query_length,
                        force_adapter_query_length,
                    )
                adapter_query_length = force_adapter_query_length
            else:
                # Auto-detect adapter query length from checkpoint if possible.
                detected_query_length = None
                if adapter_ckpt_dir is not None:
                    ckpt_path = os.path.join(adapter_ckpt_dir, "adapter_pytorch_model.bin")
                    if os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                        if isinstance(ckpt, dict):
                            if "state_dict" in ckpt:
                                ckpt = ckpt["state_dict"]
                            elif "model_state_dict" in ckpt:
                                ckpt = ckpt["model_state_dict"]
                        if isinstance(ckpt, dict) and "decoder_query" in ckpt:
                            detected_query_length = ckpt["decoder_query"].shape[1]
                if detected_query_length is not None and detected_query_length != adapter_query_length:
                    logger.info("Adapter query_length override: %s -> %s", adapter_query_length, detected_query_length)
                    adapter_query_length = detected_query_length

            device_id = device.index if device.type == "cuda" and device.index is not None else 0
            self.adapter = DM_Adapter(
                in_channels=adapter_in_channels,
                out_channels=adapter_out_channels,
                learnable_query_length=adapter_query_length,
                num_encoder_layers=adapter_num_encoder_layers,
                num_decoder_layers=adapter_num_decoder_layers,
                ff_mult=adapter_ff_mult,
                TRAINABLE_PRECISION=precision_dtype,
                device_id=device_id,
                rank=0,
                dit_fsdp=False,
                use_usp=False,
                load_ckpt_dir=adapter_ckpt_dir,
            )

            if adapter_ckpt_dir is not None:
                self.adapter.load_ckpt()

            self.adapter_output_norm = nn.LayerNorm(adapter_out_channels, eps=eps)
            # FSDP does not support scalar (0D) parameters.
            self.adapter_output_gate = nn.Parameter(torch.tensor([1e-3], dtype=precision_dtype))

            self.resampler = SanaBridgeResampler(
                in_dim=adapter_out_channels,
                out_dim=caption_channels,
                num_queries=num_prompt_queries,
                num_heads=resampler_num_heads,
                mlp_mult=resampler_mlp_mult,
                dropout=0.0,
            )
            self.projector = None

        self.to(device=device, dtype=precision_dtype)

        # Optional LoRA on SmolVLM2 text tower
        if lora_enable:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "v_proj"]
            if not self.use_vision_head:
                # Stage-3 text-only runs do not route through vision tower; excluding these modules
                # keeps DDP on the fast path without find_unused_parameters.
                if lora_exclude_patterns is None:
                    lora_exclude_patterns = ["vision_model"]
                elif "vision_model" not in lora_exclude_patterns:
                    lora_exclude_patterns = list(lora_exclude_patterns) + ["vision_model"]
            self._apply_lora_to_smolvlm2(
                target_modules=lora_target_modules,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                include_patterns=lora_include_patterns,
                exclude_patterns=lora_exclude_patterns,
            )

    def _apply_lora_to_smolvlm2(
        self,
        target_modules: List[str],
        r: int,
        alpha: int,
        dropout: float,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        import torch.nn.functional as F

        class LoRALinear(nn.Module):
            def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
                super().__init__()
                self.in_features = base.in_features
                self.out_features = base.out_features
                self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
                self.bias = None
                if base.bias is not None:
                    self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
                self.r = int(r)
                self.alpha = int(alpha)
                self.scaling = float(alpha) / float(r) if r > 0 else 1.0
                self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=base.weight.device, dtype=base.weight.dtype))
                self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype))
                self.dropout = float(dropout)
                nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
                nn.init.zeros_(self.lora_B)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                result = F.linear(x, self.weight, self.bias)
                if self.r > 0:
                    lora_input = F.dropout(x, p=self.dropout, training=self.training)
                    update = F.linear(lora_input, self.lora_A)
                    update = F.linear(update, self.lora_B)
                    result = result + update * self.scaling
                return result

        def _replace(module: nn.Module, name: str, new_module: nn.Module):
            for attr in dir(module):
                if getattr(module, attr, None) is new_module:
                    setattr(module, attr, new_module)
                    return True
            if hasattr(module, "_modules") and name in module._modules:
                module._modules[name] = new_module
                return True
            return False

        target_set = set(target_modules)
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []
        model = getattr(self.smolvlm2_model, "_model", self.smolvlm2_model)
        replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if include_patterns and not any(pat in name for pat in include_patterns):
                continue
            if exclude_patterns and any(pat in name for pat in exclude_patterns):
                continue
            if any(name.endswith(t) for t in target_set):
                parent_path = name.split(".")[:-1]
                leaf = name.split(".")[-1]
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                lora_layer.to(module.weight.device, dtype=module.weight.dtype)
                _replace(parent, leaf, lora_layer)
                replaced += 1

        # Freeze all base params; keep LoRA trainable
        for p in model.parameters():
            p.requires_grad = False
        for m in model.modules():
            if hasattr(m, "lora_A") or hasattr(m, "lora_B"):
                for p in m.parameters():
                    if p.requires_grad:
                        continue
                if hasattr(m, "lora_A"):
                    m.lora_A.requires_grad = True
                if hasattr(m, "lora_B"):
                    m.lora_B.requires_grad = True
        lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            (
                "Applied LoRA to SmolVLM2: replaced_linear_layers=%d target_modules=%s "
                "trainable_lora_params=%d include_patterns=%s exclude_patterns=%s"
            ),
            replaced,
            sorted(target_set),
            lora_trainable,
            include_patterns,
            exclude_patterns,
        )

    def _get_tokenizer(self):
        tokenizer = self.smolvlm2_model.get_tokenizer()
        if tokenizer is None:
            if not hasattr(self, "_cached_tokenizer"):
                logger.warning("Tokenizer not found in SmolVLM2 checkpoint, loading from HuggingFace...")
                try:
                    from transformers import AutoTokenizer
                    self._cached_tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_model_id,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception as exc:
                    vocab_size = 32000
                    cfg = getattr(self.smolvlm2_model, "config", None)
                    if cfg is not None and hasattr(cfg, "vocab_size"):
                        vocab_size = int(cfg.vocab_size)
                    if hasattr(self.smolvlm2_model, "_model") and hasattr(self.smolvlm2_model._model, "config"):
                        vocab_size = int(getattr(self.smolvlm2_model._model.config, "vocab_size", vocab_size))
                    logger.warning(
                        "Failed to load tokenizer locally (%s). Falling back to SimpleTokenizer(vocab_size=%s).",
                        exc,
                        vocab_size,
                    )
                    self._cached_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
            tokenizer = self._cached_tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        return tokenizer

    def encode_prompts(
        self,
        prompts: List[str],
        return_mask: bool = False,
        return_all_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        tokenizer = self._get_tokenizer()
        prompts = [
            p if p is not None and str(p).strip() else (tokenizer.eos_token or tokenizer.pad_token or " ")
            for p in prompts
        ]
        tokenize_max_length = int(self.max_length)
        tokenize_padding = True
        if self.strict_sana_parity_text_path:
            if self.sana_model_max_length < 1:
                raise RuntimeError(f"Invalid sana_model_max_length={self.sana_model_max_length}")
            has_chi_prefix = bool(self.sana_chi_prompt) and all(str(p).startswith(self.sana_chi_prompt) for p in prompts)
            if self.strict_sana_use_full_text_window:
                tokenize_max_length = int(self.max_length)
            elif has_chi_prefix:
                if self._chi_prompt_token_count is None:
                    self._chi_prompt_token_count = len(tokenizer.encode(self.sana_chi_prompt))
                tokenize_max_length = int(self._chi_prompt_token_count + self.sana_model_max_length - 2)
                tokenize_max_length = max(tokenize_max_length, self.sana_model_max_length)
            else:
                tokenize_max_length = int(self.sana_model_max_length)
            if tokenize_max_length > int(self.max_length):
                raise RuntimeError(
                    "Strict SANA-parity requires max_length >= tokenize_max_length, "
                    f"got max_length={self.max_length}, tokenize_max_length={tokenize_max_length}"
                )
            tokenize_padding = "max_length"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=tokenize_padding,
            truncation=True,
            max_length=tokenize_max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Keep frozen-text execution cheap, but allow autograd when LoRA is trainable.
        smol_has_trainable = any(p.requires_grad for p in self.smolvlm2_model.parameters())
        need_lora_grad = bool(self.training and smol_has_trainable)
        prev_train = bool(self.smolvlm2_model.training)
        if need_lora_grad and not prev_train:
            # SmolVLMModel.forward uses no_grad() when not training.
            # Temporarily switch to train mode so LoRA receives gradients.
            self.smolvlm2_model.train(True)
        grad_ctx = torch.enable_grad() if need_lora_grad else torch.no_grad()
        try:
            with grad_ctx:
                outputs = self.smolvlm2_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            if need_lora_grad and not prev_train:
                self.smolvlm2_model.train(False)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)):
            hidden_states = outputs.hidden_states[-1]
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            hidden_states = outputs[0]
        else:
            raise RuntimeError("SmolVLM2 output does not contain hidden states")

        all_hidden_states = None
        if return_all_hidden_states and hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)):
            # Drop embedding output if present by keeping only tensors with same ndim as token hidden states.
            all_hidden_states = [h for h in outputs.hidden_states if isinstance(h, torch.Tensor) and h.dim() == 3]
        if self.strict_sana_parity_text_path:
            target_len = int(self.sana_model_max_length)
            cur_len = int(hidden_states.shape[1])
            if cur_len < target_len:
                raise RuntimeError(
                    f"Strict SANA-parity requires hidden length >= {target_len}, got {cur_len}."
                )
            if cur_len != target_len:
                select_index = build_strict_sana_select_index(
                    cur_len,
                    target_len,
                    strategy=self.strict_sana_token_select_strategy,
                    head_tokens=self.strict_sana_head_tokens,
                    tail_tokens=self.strict_sana_tail_tokens,
                    device=hidden_states.device,
                )
                hidden_states = hidden_states.index_select(1, select_index)
                if attention_mask is not None:
                    attention_mask = attention_mask.index_select(1, select_index.to(attention_mask.device))
                if all_hidden_states is not None:
                    all_hidden_states = [h.index_select(1, select_index.to(h.device)) for h in all_hidden_states]
        if return_mask:
            return hidden_states, attention_mask, all_hidden_states
        return hidden_states, None, all_hidden_states

    def forward(self, prompts: List[str], return_mask: bool = False, return_aux: bool = False):
        need_all = self.projector_type in ("mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck")
        hidden_states, attention_mask, all_hidden_states = self.encode_prompts(
            prompts,
            return_mask=return_mask,
            return_all_hidden_states=need_all,
        )
        aux: Dict[str, Any] = {}

        if need_all:
            if all_hidden_states is None:
                raise RuntimeError("MCP projector requires output hidden_states, got None")
            if return_aux and len(all_hidden_states) > 0:
                aux["hidden0"] = all_hidden_states[0]
            prompt_embeds = self.projector(all_hidden_states, temperature=self.mcp_fusion_temperature)
        else:
            if self.use_vision_head:
                vision_tokens = self.smolvlm2_vision_head(hidden_states)
                adapter_input = vision_tokens
            else:
                adapter_input = hidden_states
            adapter_output = self.adapter(adapter_input)

            if adapter_output.dim() == 2:
                adapter_output = adapter_output.unsqueeze(0)

            adapter_output = self.adapter_output_norm(adapter_output)
            if self.gate_min_value > 0.0:
                gate = torch.clamp(self.adapter_output_gate, min=self.gate_min_value)
            else:
                gate = self.adapter_output_gate
            adapter_output = adapter_output * gate
            prompt_embeds = self.resampler(adapter_output)

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        if return_mask:
            if attention_mask is not None and attention_mask.shape[1] == prompt_embeds.shape[1]:
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                prompt_mask = attention_mask.to(device=prompt_embeds.device, dtype=torch.long)
            else:
                if self.fail_fast_mask:
                    raise RuntimeError(
                        "Fail-fast mask enabled: attention_mask/prompt_embeds shape mismatch: "
                        f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None}, "
                        f"prompt_embeds={tuple(prompt_embeds.shape)}"
                    )
                # TODO(cleanup): tighten this fallback to an explicit error in strict parity mode.
                # All-ones mask is robust for runtime, but can hide token-length/mask mismatches.
                if not hasattr(self, "_mask_mismatch_warned"):
                    logger.warning(
                        "Prompt mask fallback to all-ones: attention_mask shape=%s, prompt_embeds shape=%s",
                        tuple(attention_mask.shape) if attention_mask is not None else None,
                        tuple(prompt_embeds.shape),
                    )
                    self._mask_mismatch_warned = True
                prompt_mask = torch.ones(
                    prompt_embeds.shape[:2],
                    device=prompt_embeds.device,
                    dtype=torch.long,
                )
            if return_aux:
                return prompt_embeds, prompt_mask, aux
            return prompt_embeds, prompt_mask
        if return_aux:
            return prompt_embeds, aux
        return prompt_embeds
