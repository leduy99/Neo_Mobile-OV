#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    build_deepspeed_config,
    cleanup_distributed,
    full_state_dict,
    rank0_print,
    scalar_mean,
    setup_distributed,
    write_deepspeed_config,
)


class PromptDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        max_prompts: int = -1,
        *,
        caption_aug: bool = True,
        caption_variant_columns: list[str] | None = None,
        caption_variant_weights: list[float] | None = None,
        caption_fallback_column: str = "caption",
    ):
        path = Path(path)
        self.caption_aug_enabled = bool(caption_aug)
        self.caption_variant_columns = caption_variant_columns or ["caption_short", "caption_medium", "caption_long"]
        self.caption_variant_weights = caption_variant_weights
        self.caption_fallback_column = str(caption_fallback_column or "caption")
        self.items: list[dict[str, object]] = []

        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            fallback_column = self.caption_fallback_column if self.caption_fallback_column in df.columns else None
            if fallback_column is None:
                fallback_column = next((c for c in ["prompt", "caption", "text"] if c in df.columns), None)
            if fallback_column is None:
                raise ValueError(f"{path} must contain one of columns: prompt, caption, text")

            for _, row in df.iterrows():
                fallback = clean_text(row.get(fallback_column, ""))
                variants: list[tuple[str, str, float]] = []
                if self.caption_aug_enabled:
                    for idx, col in enumerate(self.caption_variant_columns):
                        if col not in df.columns:
                            continue
                        text = clean_text(row.get(col, ""))
                        if not text:
                            continue
                        weight = 1.0
                        if self.caption_variant_weights is not None and idx < len(self.caption_variant_weights):
                            weight = float(self.caption_variant_weights[idx])
                        variants.append((text, col, weight))
                if not variants and fallback:
                    variants.append((fallback, fallback_column, 1.0))
                if variants:
                    self.items.append({"fallback": fallback, "variants": variants})
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                prompt = clean_text(line)
                if prompt:
                    self.items.append({"fallback": prompt, "variants": [(prompt, "text", 1.0)]})
        if max_prompts > 0:
            self.items = self.items[:max_prompts]
        if not self.items:
            raise ValueError(f"No prompts found in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> str:
        item = self.items[index]
        variants = item["variants"]
        assert isinstance(variants, list)
        if len(variants) == 1:
            return str(variants[0][0])
        weights = [float(v[2]) for v in variants]
        return str(random.choices(variants, weights=weights, k=1)[0][0])


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


def split_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_float_list(value: str | None) -> list[float] | None:
    parts = split_csv_list(value)
    if not parts:
        return None
    return [float(part) for part in parts]


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


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


def masked_mean_norm(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    token_norm = tokens.norm(dim=-1, keepdim=True)
    return (token_norm * weights).sum(dim=1) / denom


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def wrap_bridge(
    bridge: MobileOVNeodragonTextBridge,
    *,
    parallel: str,
    device: torch.device,
    local_rank: int,
) -> torch.nn.Module:
    parallel = parallel.lower()
    if parallel == "none":
        return bridge
    if parallel == "ddp":
        return DDP(bridge, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if parallel == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        strategy = ShardingStrategy.FULL_SHARD if torch.distributed.get_world_size() > 1 else ShardingStrategy.NO_SHARD
        return FSDP(
            bridge,
            device_id=device,
            use_orig_params=True,
            sharding_strategy=strategy,
            sync_module_states=False,
        )
    if parallel == "deepspeed":
        return bridge
    raise ValueError(f"Unknown --parallel={parallel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--prompts", required=True, help="Text file or CSV/TSV with prompt/caption/text column.")
    parser.add_argument("--output-dir", default="output/neodragon_text_bridge_train")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument(
        "--caption-aug",
        action=argparse.BooleanOptionalAction,
        default=env_flag("MOBILEOV_CAPTION_AUG", True),
        help="Randomly sample caption variants from recaption CSV columns.",
    )
    parser.add_argument(
        "--caption-variant-columns",
        default=os.environ.get("MOBILEOV_CAPTION_AUG_COLUMNS", "caption_short,caption_medium,caption_long"),
    )
    parser.add_argument(
        "--caption-variant-weights",
        default=os.environ.get("MOBILEOV_CAPTION_AUG_WEIGHTS", ""),
        help="Optional comma-separated weights matching --caption-variant-columns.",
    )
    parser.add_argument(
        "--caption-fallback-column",
        default=os.environ.get("MOBILEOV_CAPTION_FALLBACK_COLUMN", "caption"),
    )
    parser.add_argument("--pooled-weight", type=float, default=0.25)
    parser.add_argument("--cos-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--append-prompt-modifier", action="store_true", default=True)
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp", "deepspeed"], default="none")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")
    ctx = setup_distributed()
    if args.parallel in {"ddp", "fsdp"} and not ctx.is_distributed:
        rank0_print(ctx, f"Warning: --parallel={args.parallel} requested with WORLD_SIZE=1; falling back to --parallel=none.")
        args.parallel = "none"

    cfg = load_config(args.config)
    if args.dtype:
        cfg.train.dtype = args.dtype
        cfg.backend.dtype = args.dtype
    frozen_dtype = dtype_from_name(cfg.backend.dtype)
    if ctx.device.type == "cpu":
        frozen_dtype = torch.float32
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    caption_columns = split_csv_list(args.caption_variant_columns)
    caption_weights = parse_float_list(args.caption_variant_weights)
    if caption_weights is not None and len(caption_weights) != len(caption_columns):
        raise ValueError(
            f"--caption-variant-weights has {len(caption_weights)} entries but "
            f"--caption-variant-columns has {len(caption_columns)} entries"
        )
    dataset = PromptDataset(
        args.prompts,
        max_prompts=args.max_prompts,
        caption_aug=args.caption_aug,
        caption_variant_columns=caption_columns,
        caption_variant_weights=caption_weights,
        caption_fallback_column=args.caption_fallback_column,
    )
    sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True) if ctx.is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    batches = cycle_loader(loader, sampler)

    rank0_print(
        ctx,
        f"Text bridge distill: parallel={args.parallel} world_size={ctx.world_size} "
        f"batch_per_gpu={args.batch_size} prompts={len(dataset)} dtype={frozen_dtype} "
        f"caption_aug={args.caption_aug} caption_columns={caption_columns} caption_weights={caption_weights}",
    )
    teacher, context_adapter, prompt_modifier = load_neodragon_text_modules(cfg, ctx.device, frozen_dtype)
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=ctx.device, dtype=frozen_dtype).train()
    bridge_model = wrap_bridge(bridge, parallel=args.parallel, device=ctx.device, local_rank=ctx.local_rank)

    deepspeed_engine = None
    opt = None
    if args.parallel == "deepspeed":
        import deepspeed

        ds_config = build_deepspeed_config(
            micro_batch_size=args.batch_size,
            world_size=ctx.world_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero_stage=args.deepspeed_zero_stage,
            dtype=frozen_dtype,
        )
        if ctx.is_main:
            write_deepspeed_config(ds_config, out_dir)
        trainable = [p for p in bridge_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
        deepspeed_engine, opt, _, _ = deepspeed.initialize(
            model=bridge_model,
            optimizer=opt,
            model_parameters=trainable,
            config=ds_config,
            dist_init_required=False if torch.distributed.is_initialized() else None,
        )
        bridge_model = deepspeed_engine
    else:
        opt = torch.optim.AdamW((p for p in bridge_model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)

    history: list[dict[str, float]] = []
    iterator = range(1, args.steps + 1)
    pbar = tqdm(iterator, desc="Train Mobile-OV Neodragon text bridge", disable=not ctx.is_main)
    for step in pbar:
        prompts_raw = [str(x) for x in next(batches)]
        prompts = [p + prompt_modifier for p in prompts_raw] if args.append_prompt_modifier else prompts_raw

        with torch.no_grad():
            target_tokens, target_mask, target_pooled = teacher(prompts, ctx.device)
            target_tokens = context_adapter(target_tokens)

        pred_tokens, pred_mask, pred_pooled = bridge_model(prompts)
        target_tokens = target_tokens.float()
        target_pooled = target_pooled.float()
        target_mask = target_mask.to(device=ctx.device)

        token_loss = masked_mse(pred_tokens.float(), target_tokens, target_mask)
        pooled_loss = F.mse_loss(pred_pooled.float(), target_pooled)
        cos_loss = masked_cosine_loss(pred_tokens.float(), target_tokens, target_mask)
        loss = token_loss + args.pooled_weight * pooled_loss + args.cos_weight * cos_loss

        if deepspeed_engine is not None:
            deepspeed_engine.backward(loss)
            deepspeed_engine.step()
        else:
            assert opt is not None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad_norm > 0:
                if args.parallel == "fsdp" and hasattr(bridge_model, "clip_grad_norm_"):
                    bridge_model.clip_grad_norm_(args.clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(bridge_model.parameters(), args.clip_grad_norm)
            opt.step()

        with torch.no_grad():
            pred_norm = masked_mean_norm(pred_tokens.float(), target_mask).mean()
            target_norm = masked_mean_norm(target_tokens, target_mask).mean()
            mask_tok = target_mask.float().sum(dim=1).mean()

        if step % args.log_every == 0 or step == 1:
            item = {
                "step": float(step),
                "loss": scalar_mean(loss.detach(), ctx),
                "token_loss": scalar_mean(token_loss.detach(), ctx),
                "pooled_loss": scalar_mean(pooled_loss.detach(), ctx),
                "cos_loss": scalar_mean(cos_loss.detach(), ctx),
                "pred_norm": scalar_mean(pred_norm.detach(), ctx),
                "target_norm": scalar_mean(target_norm.detach(), ctx),
                "target_mask_tokens": scalar_mean(mask_tok.detach(), ctx),
                "world_size": float(ctx.world_size),
            }
            if ctx.is_main:
                history.append(item)
                pbar.set_postfix(
                    loss=f"{item['loss']:.4f}",
                    tok=f"{item['token_loss']:.4f}",
                    pool=f"{item['pooled_loss']:.4f}",
                    cos=f"{item['cos_loss']:.4f}",
                    norm=f"{item['pred_norm']:.1f}/{item['target_norm']:.1f}",
                )

        if step % args.save_every == 0 or step == args.steps:
            state = full_state_dict(bridge_model, args.parallel)
            if ctx.is_main:
                torch.save(
                    {
                        "step": step,
                        "bridge": state,
                        "config": cfg,
                        "args": vars(args),
                        "history": history,
                        "target": "neodragon_dit_condition_direct",
                        "parallel": {
                            "backend": args.parallel,
                            "world_size": ctx.world_size,
                            "deepspeed_zero_stage": args.deepspeed_zero_stage if args.parallel == "deepspeed" else None,
                        },
                        "shapes": {
                            "prompt_embeds": [None, 128, 1536],
                            "prompt_mask": [None, 128],
                            "pooled_prompt_embeds": [None, 2048],
                        },
                    },
                    out_dir / "neodragon_text_bridge_latest.pt",
                )
                (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            barrier()

    rank0_print(ctx, f"Saved bridge to {out_dir / 'neodragon_text_bridge_latest.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
