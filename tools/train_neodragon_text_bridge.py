#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from new_mobile_ov.training.neodragon_objectives import (
    flat_cosine_distance,
    linear_ramp,
    masked_mean_norm,
    masked_token_cosine,
    masked_token_mse,
    pooled_cosine,
    relational_cosine,
    token_norm_alignment,
)
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


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def load_neodragon_functional_modules(cfg, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import DIT_ID
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler

    dit = PyramidMMDiT.from_pretrained(f"{local_model_path}/{DIT_ID}", torch_dtype=dtype).to(device).eval()
    for param in dit.parameters():
        param.requires_grad_(False)
    return dit, PyramidFlowMatchEulerDiscreteScheduler()


def sample_functional_input(cfg, scheduler, batch_size: int, device: torch.device, dtype: torch.dtype):
    from neodragon.utils.generation_utils import _get_pyramid_latent, _prepare_past_condition_latents

    latent_t = ((int(cfg.data.frame_num) - 1) // 8) + 1
    latent_h = int(cfg.data.height) // 8
    latent_w = int(cfg.data.width) // 8
    latents = torch.randn(
        batch_size,
        16,
        latent_t,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
    )
    unit_index = int(torch.randint(1, latent_t, (1,), device=device).item())
    stage = int(torch.randint(0, scheduler.config.stages, (1,), device=device).item())
    past_units = [latents[:, :, i : i + 1] for i in range(unit_index)]
    past_conditions = _prepare_past_condition_latents(
        past_units,
        num_stages=scheduler.config.stages,
        do_classifier_free_guidance=False,
    )
    clean = _get_pyramid_latent(latents[:, :, unit_index : unit_index + 1], scheduler.config.stages)[stage]
    noise = torch.randn_like(clean)
    t_idx = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    sigmas = scheduler.sigmas_per_stage[stage].to(device=device, dtype=dtype)[t_idx]
    timestep = scheduler.timesteps_per_stage[stage].to(device=device, dtype=dtype)[t_idx]
    while sigmas.dim() < clean.dim():
        sigmas = sigmas.view(-1, *([1] * (clean.dim() - 1)))
    noisy = sigmas * noise + (1.0 - sigmas) * clean
    return past_conditions[stage] + [noisy], timestep, stage, unit_index


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
            sync_module_states=True,
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
    parser.add_argument("--raw-token-weight", type=float, default=1.0)
    parser.add_argument("--normalized-token-weight", type=float, default=0.0)
    parser.add_argument("--cos-weight", type=float, default=0.05)
    parser.add_argument("--token-norm-weight", type=float, default=0.0)
    parser.add_argument("--pooled-weight", type=float, default=0.25)
    parser.add_argument("--pooled-cos-weight", type=float, default=0.0)
    parser.add_argument("--relational-weight", type=float, default=0.0)
    parser.add_argument("--functional-weight", type=float, default=0.0)
    parser.add_argument("--functional-cos-weight", type=float, default=0.0)
    parser.add_argument("--functional-start-step", type=int, default=1)
    parser.add_argument("--functional-ramp-steps", type=int, default=0)
    parser.add_argument("--functional-every", type=int, default=1)
    parser.add_argument("--functional-batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--append-prompt-modifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp", "deepspeed"], default="none")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--keep-step-checkpoints",
        action="store_true",
        default=env_flag("KEEP_STEP_CHECKPOINTS", False),
        help="Also save neodragon_text_bridge_stepXXXXXX.pt at each save interval.",
    )
    args = parser.parse_args()

    weighted_losses = {
        "raw token": args.raw_token_weight,
        "normalized token": args.normalized_token_weight,
        "token cosine": args.cos_weight,
        "token norm": args.token_norm_weight,
        "pooled": args.pooled_weight,
        "pooled cosine": args.pooled_cos_weight,
        "relational": args.relational_weight,
        "functional": args.functional_weight,
        "functional cosine": args.functional_cos_weight,
    }
    invalid_weights = {name: value for name, value in weighted_losses.items() if value < 0.0}
    if invalid_weights:
        raise ValueError(f"Loss weights must be non-negative, got {invalid_weights}")
    if not any(value > 0.0 for value in weighted_losses.values()):
        raise ValueError("At least one bridge distillation loss weight must be positive.")
    if args.functional_every < 1 or args.functional_batch_size < 1:
        raise ValueError("--functional-every and --functional-batch-size must be >= 1.")
    if args.steps < 1 or args.log_every < 1 or args.save_every < 1:
        raise ValueError("--steps, --log-every, and --save-every must be >= 1.")

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
    functional_dit = None
    functional_scheduler = None
    functional_enabled = args.functional_weight > 0.0 or args.functional_cos_weight > 0.0
    if functional_enabled:
        functional_dit, functional_scheduler = load_neodragon_functional_modules(
            cfg,
            ctx.device,
            frozen_dtype,
        )
        rank0_print(
            ctx,
            "Functional bridge distillation enabled: "
            f"mse={args.functional_weight} cos={args.functional_cos_weight} "
            f"start={args.functional_start_step} ramp={args.functional_ramp_steps} "
            f"every={args.functional_every} batch={args.functional_batch_size}",
        )
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

        raw_token_loss = masked_token_mse(pred_tokens, target_tokens, target_mask)
        normalized_token_loss = masked_token_mse(
            pred_tokens,
            target_tokens,
            target_mask,
            normalize_tokens=True,
        )
        pooled_loss = F.mse_loss(pred_pooled.float(), target_pooled)
        cos_loss = masked_token_cosine(pred_tokens, target_tokens, target_mask)
        norm_loss = token_norm_alignment(pred_tokens, target_tokens, target_mask)
        pooled_cos_loss = pooled_cosine(pred_pooled, target_pooled)
        relational_loss = relational_cosine(pred_tokens, target_tokens, target_mask)

        functional_loss = pred_tokens.new_zeros(())
        functional_cos_loss = pred_tokens.new_zeros(())
        functional_scale = 0.0
        functional_stage = -1
        functional_unit = -1
        run_functional = bool(
            functional_enabled
            and functional_dit is not None
            and functional_scheduler is not None
            and step >= args.functional_start_step
            and (step - args.functional_start_step) % max(args.functional_every, 1) == 0
        )
        if run_functional:
            functional_scale = linear_ramp(
                step,
                start_step=args.functional_start_step,
                ramp_steps=args.functional_ramp_steps,
            )
            effect_bs = min(max(args.functional_batch_size, 1), pred_tokens.shape[0])
            stage_input, timestep, functional_stage, functional_unit = sample_functional_input(
                cfg,
                functional_scheduler,
                effect_bs,
                ctx.device,
                frozen_dtype,
            )
            with torch.no_grad():
                teacher_prediction = functional_dit(
                    sample=[stage_input],
                    encoder_hidden_states=target_tokens[:effect_bs].to(dtype=frozen_dtype),
                    encoder_attention_mask=target_mask[:effect_bs],
                    pooled_projections=target_pooled[:effect_bs].to(dtype=frozen_dtype),
                    timestep_ratio=timestep,
                )[0]
            student_prediction = functional_dit(
                sample=[stage_input],
                encoder_hidden_states=pred_tokens[:effect_bs],
                encoder_attention_mask=pred_mask[:effect_bs],
                pooled_projections=pred_pooled[:effect_bs],
                timestep_ratio=timestep,
            )[0]
            functional_loss = F.mse_loss(student_prediction.float(), teacher_prediction.float())
            functional_cos_loss = flat_cosine_distance(student_prediction, teacher_prediction)

        loss = (
            args.raw_token_weight * raw_token_loss
            + args.normalized_token_weight * normalized_token_loss
            + args.cos_weight * cos_loss
            + args.token_norm_weight * norm_loss
            + args.pooled_weight * pooled_loss
            + args.pooled_cos_weight * pooled_cos_loss
            + args.relational_weight * relational_loss
            + functional_scale
            * (
                args.functional_weight * functional_loss
                + args.functional_cos_weight * functional_cos_loss
            )
        )

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
            mask_accuracy = (pred_mask.bool() == target_mask.bool()).float().mean()

        if step % args.log_every == 0 or step == 1:
            item = {
                "step": float(step),
                "loss": scalar_mean(loss.detach(), ctx),
                "raw_token_loss": scalar_mean(raw_token_loss.detach(), ctx),
                "normalized_token_loss": scalar_mean(normalized_token_loss.detach(), ctx),
                "pooled_loss": scalar_mean(pooled_loss.detach(), ctx),
                "cos_loss": scalar_mean(cos_loss.detach(), ctx),
                "norm_loss": scalar_mean(norm_loss.detach(), ctx),
                "pooled_cos_loss": scalar_mean(pooled_cos_loss.detach(), ctx),
                "relational_loss": scalar_mean(relational_loss.detach(), ctx),
                "mask_accuracy": scalar_mean(mask_accuracy.detach(), ctx),
                "functional_loss": scalar_mean(functional_loss.detach(), ctx),
                "functional_cos_loss": scalar_mean(functional_cos_loss.detach(), ctx),
                "functional_scale": float(functional_scale),
                "functional_stage": float(functional_stage),
                "functional_unit": float(functional_unit),
                "pred_norm": scalar_mean(pred_norm.detach(), ctx),
                "target_norm": scalar_mean(target_norm.detach(), ctx),
                "target_mask_tokens": scalar_mean(mask_tok.detach(), ctx),
                "world_size": float(ctx.world_size),
            }
            if ctx.is_main:
                history.append(item)
                pbar.set_postfix(
                    loss=f"{item['loss']:.4f}",
                    tok=f"{item['raw_token_loss']:.4f}",
                    pool=f"{item['pooled_loss']:.4f}",
                    cos=f"{item['cos_loss']:.4f}",
                    func=f"{item['functional_loss']:.4f}",
                    norm=f"{item['pred_norm']:.1f}/{item['target_norm']:.1f}",
                )

        if step % args.save_every == 0 or step == args.steps:
            state = full_state_dict(bridge_model, args.parallel)
            if ctx.is_main:
                payload = {
                    "step": step,
                    "bridge": state,
                    "config": cfg,
                    "args": vars(args),
                    "history": history,
                    "target": "neodragon_dit_condition_direct",
                    "architecture": {
                        "bridge_contract": "original_neodragon_direct_condition",
                        "functional_distillation": functional_enabled,
                    },
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
                }
                torch.save(payload, out_dir / "neodragon_text_bridge_latest.pt")
                if args.keep_step_checkpoints:
                    torch.save(payload, out_dir / f"neodragon_text_bridge_step{step:06d}.pt")
                (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            barrier()

    rank0_print(ctx, f"Saved bridge to {out_dir / 'neodragon_text_bridge_latest.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
