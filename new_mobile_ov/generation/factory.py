from __future__ import annotations

import torch

from new_mobile_ov.config import BackendConfig
from new_mobile_ov.generation.backends.base import AnchorGenerationBackend
from new_mobile_ov.generation.backends.mobile_o_sana_0_5b import MobileOSana05BBackend
from new_mobile_ov.generation.backends.mobile_ov_neodragon import MobileOVNeodragonBackend
from new_mobile_ov.generation.backends.mobile_ov_current import MobileOVCurrentBackend


def _dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def build_generation_backend(cfg: BackendConfig, device: torch.device) -> AnchorGenerationBackend:
    if cfg.name == "mobile_ov_current":
        return MobileOVCurrentBackend(checkpoint_path=cfg.checkpoint_path, device=device)
    if cfg.name == "mobile_o_sana_0_5b":
        if not cfg.model_path:
            raise ValueError("backend.model_path is required for mobile_o_sana_0_5b")
        return MobileOSana05BBackend(model_path=cfg.model_path, device=device, dtype=_dtype(cfg.dtype))
    if cfg.name == "mobile_ov_neodragon":
        repo_path = cfg.extra.get("repo_path") if cfg.extra else None
        cache_dir = cfg.extra.get("cache_dir") if cfg.extra else None
        mode = cfg.extra.get("mode", "hybrid") if cfg.extra else "hybrid"
        model_id = cfg.extra.get("model_id", "karnewar/Neodragon") if cfg.extra else "karnewar/Neodragon"
        repo_url = cfg.extra.get("repo_url") if cfg.extra else None
        if not repo_path:
            raise ValueError("backend.extra.repo_path is required for mobile_ov_neodragon")
        if not cache_dir:
            raise ValueError("backend.extra.cache_dir is required for mobile_ov_neodragon")
        return MobileOVNeodragonBackend(
            repo_path=repo_path,
            cache_dir=cache_dir,
            device=device,
            dtype=_dtype(cfg.dtype),
            mode=mode,
            model_id=model_id,
            repo_url=repo_url,
        )
    raise ValueError(f"Unknown generation backend: {cfg.name}")
