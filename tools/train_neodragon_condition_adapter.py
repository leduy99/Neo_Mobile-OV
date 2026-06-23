#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.generation.neodragon_condition_adapter import BridgeToNeodragonConditionAdapter


class PromptDataset(Dataset):
    def __init__(self, path: str | Path, max_prompts: int = -1):
        path = Path(path)
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            column = next((c for c in ["prompt", "caption", "text"] if c in df.columns), None)
            if column is None:
                raise ValueError(f"{path} must contain one of columns: prompt, caption, text")
            prompts = [str(x) for x in df[column].dropna().tolist()]
        else:
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_prompts > 0:
            prompts = prompts[:max_prompts]
        if not prompts:
            raise ValueError(f"No prompts found in {path}")
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def load_neodragon_text_modules(cfg, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import CONTEXT_ADAPTER_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    for module in [text_bundle, context_adapter]:
        for param in module.parameters():
            param.requires_grad_(False)
    return text_bundle, context_adapter, DEFAULT_PROMPT_MODIFIER


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * weights).sum() / denom


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    weights = mask.to(device=pred.device, dtype=cos.dtype)
    return 1.0 - (cos * weights).sum() / weights.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--prompts", required=True, help="Text file or CSV/TSV with prompt/caption/text column.")
    parser.add_argument("--output-dir", default="output/neodragon_condition_adapter_train")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument("--pooled-weight", type=float, default=0.25)
    parser.add_argument("--cos-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")

    cfg = load_config(args.config)
    if args.dtype:
        cfg.train.dtype = args.dtype
        cfg.backend.dtype = args.dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen_dtype = dtype_from_name(cfg.backend.dtype)
    if device.type == "cpu":
        frozen_dtype = torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PromptDataset(args.prompts, max_prompts=args.max_prompts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batches = itertools.cycle(loader)

    bridge = MobileOVTextBridge(cfg.bridge, device=device, dtype=frozen_dtype).eval()
    for param in bridge.parameters():
        param.requires_grad_(False)
    text_bundle, context_adapter, default_prompt_modifier = load_neodragon_text_modules(cfg, device, frozen_dtype)

    adapter = BridgeToNeodragonConditionAdapter(
        bridge_dim=cfg.bridge.caption_channels,
        neodragon_dim=1536,
        pooled_dim=2048,
        num_queries=128,
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.0)

    history: list[dict[str, float]] = []
    pbar = tqdm(range(1, args.steps + 1), desc="Train bridge->Neodragon condition")
    for step in pbar:
        prompts = [str(x) for x in next(batches)]
        neodragon_prompts = [p + default_prompt_modifier for p in prompts]
        with torch.no_grad():
            bridge_tokens, bridge_mask, _ = bridge.encode(prompts)
            target_tokens, target_mask, target_pooled = text_bundle(neodragon_prompts, device)
            target_tokens = context_adapter(target_tokens)

        pred_tokens, pred_mask, pred_pooled = adapter(bridge_tokens.float(), bridge_mask)
        target_tokens = target_tokens.float()
        target_pooled = target_pooled.float()
        target_mask = target_mask.to(device=device)

        token_loss = masked_mse(pred_tokens, target_tokens, target_mask)
        pooled_loss = F.mse_loss(pred_pooled.float(), target_pooled)
        cos_loss = masked_cosine_loss(pred_tokens, target_tokens, target_mask)
        loss = token_loss + args.pooled_weight * pooled_loss + args.cos_weight * cos_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()

        item = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "token_loss": float(token_loss.detach().cpu()),
            "pooled_loss": float(pooled_loss.detach().cpu()),
            "cos_loss": float(cos_loss.detach().cpu()),
        }
        if step % args.log_every == 0:
            history.append(item)
            pbar.set_postfix(
                loss=f"{item['loss']:.4f}",
                tok=f"{item['token_loss']:.4f}",
                pool=f"{item['pooled_loss']:.4f}",
                cos=f"{item['cos_loss']:.4f}",
            )
        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "adapter": adapter.state_dict(),
                    "config": cfg,
                    "args": vars(args),
                    "history": history,
                },
                out_dir / "neodragon_condition_adapter_latest.pt",
            )
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved adapter to {out_dir / 'neodragon_condition_adapter_latest.pt'}")


if __name__ == "__main__":
    main()
