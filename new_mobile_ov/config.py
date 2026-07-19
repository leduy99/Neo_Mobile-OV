from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


GenerationBackendName = Literal["mobile_ov_current", "mobile_o_sana_0_5b", "mobile_ov_neodragon"]


@dataclass
class BridgeConfig:
    smolvlm2_ckpt_path: str = "checkpoints/smolvlm2_500m/smolvlm2_500m.pt"
    adapter_ckpt_dir: Optional[str] = None
    tokenizer_model_id: str = "HuggingFaceTB/SmolVLM-Instruct"
    max_length: int = 512
    sana_model_max_length: int = 300
    caption_channels: int = 2304
    projector_type: str = "mcp_lexical_gated"
    mcp_num_fuse_layers: int = 2
    mcp_use_refine: bool = True
    mcp_refine_kernel_size: int = 3
    mcp_lexical_gate_init: float = 0.2
    strict_sana_parity_text_path: bool = True
    strict_sana_use_full_text_window: bool = True
    strict_sana_token_select_strategy: str = "head_uniform_tail"
    strict_sana_head_tokens: int = 96
    strict_sana_tail_tokens: int = 96
    fail_fast_mask: bool = True
    neodragon_v2_conditioning: bool = False
    neodragon_resampler_heads: int = 12
    neodragon_condition_bottleneck_dim: int = 768
    neodragon_condition_scale_init: float = 0.78
    neodragon_mask_length_delta_ratio: float = 0.25
    neodragon_mask_temperature: float = 1.0


@dataclass
class BackendConfig:
    name: GenerationBackendName = "mobile_ov_current"
    checkpoint_path: Optional[str] = None
    model_path: Optional[str] = None
    dtype: str = "bf16"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionConfig:
    latent_channels: int = 16
    text_dim: int = 2304
    hidden_dim: int = 512
    temporal_len: int = 21
    num_blocks: int = 10
    mlp_ratio: int = 4
    temporal_kernel: int = 3
    temporal_dilation_cycle: list[int] = field(default_factory=lambda: [1, 2, 4])
    use_cross_attn: bool = False


@dataclass
class DataConfig:
    latent_manifest: str = "data/lmw_smoke/manifest.csv"
    output_dir: str = "output/lmw_projection_smoke"
    frame_num: int = 81
    height: int = 480
    width: int = 832


@dataclass
class TrainConfig:
    batch_size: int = 1
    lr: float = 1e-4
    total_steps: int = 1000
    motion_weight: float = 0.5
    accel_weight: float = 0.0
    log_every: int = 10
    save_every: int = 500
    dtype: str = "bf16"


@dataclass
class NewMobileOVConfig:
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key {type(obj).__name__}.{key}")
        cur = getattr(obj, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(cur, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> NewMobileOVConfig:
    cfg = NewMobileOVConfig()
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _update_dataclass(cfg, values)
