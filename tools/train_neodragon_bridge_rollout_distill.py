#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    full_state_dict,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.neodragon_objectives import (
    bridge_representation_losses,
    masked_mean_norm,
    weighted_loss_sum,
)
from new_mobile_ov.training.neodragon_rollout import rollout_distillation_loss
from tools.train_neodragon_dit_bridge import (
    VideoPromptDataset,
    collate_video_batch,
)
from tools.train_neodragon_text_bridge import (
    cycle_loader,
    load_neodragon_functional_modules,
    load_neodragon_text_modules,
)


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def promote_trainable_parameters_to_fp32(module: torch.nn.Module) -> None:
    """Keep frozen inference weights in BF16 and optimizer-owned weights in FP32."""
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def stage_log(ctx, message: str) -> None:
    print(f"[setup rank={ctx.rank}/{ctx.world_size}] {message}", flush=True)


def load_bridge(
    cfg,
    checkpoint: Path,
    device: torch.device,
    inference_dtype: torch.dtype,
) -> tuple[MobileOVNeodragonTextBridge, dict]:
    bridge = MobileOVNeodragonTextBridge(
        cfg.bridge,
        device=device,
        dtype=inference_dtype,
    ).train()
    promote_trainable_parameters_to_fp32(bridge)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("bridge", payload.get("student_state", payload))
    bridge.load_state_dict(state, strict=True)
    return bridge, payload


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def learning_rate_scale(
    global_step: int,
    *,
    initial_step: int,
    target_step: int,
    warmup_steps: int,
    final_scale: float,
) -> float:
    progress_step = global_step - initial_step
    total = max(target_step - initial_step, 1)
    if warmup_steps > 0 and progress_step <= warmup_steps:
        return 0.2 + 0.8 * progress_step / float(warmup_steps)
    denominator = max(total - warmup_steps, 1)
    progress = min(max((progress_step - warmup_steps) / denominator, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_scale) + (1.0 - float(final_scale)) * cosine


def rollout_weight_scale(
    global_step: int,
    *,
    initial_step: int,
    ramp_steps: int,
) -> float:
    if ramp_steps <= 0:
        return 1.0
    return min(max((global_step - initial_step) / float(ramp_steps), 0.0), 1.0)


def save_checkpoint(
    *,
    bridge_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parallel: str,
    output_dir: Path,
    global_step: int,
    initial_step: int,
    init_checkpoint: str,
    cfg,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    ctx,
    archive: bool,
) -> None:
    state = full_state_dict(bridge_model, parallel)
    if ctx.is_main:
        payload = {
            "step": int(global_step),
            "initial_step": int(initial_step),
            "bridge": state,
            "optimizer": optimizer.state_dict(),
            "init_checkpoint": init_checkpoint,
            "config": cfg,
            "args": vars(args),
            "history": history,
            "target": "neodragon_hybrid_full_rollout_condition",
            "architecture": {
                "bridge_contract": "original_neodragon_direct_condition",
                "dit": "released_hybrid_frozen",
                "rollout_calls": int(args.generated_units * args.num_stages),
                "differentiable_cross_unit_history": True,
                "teacher_state_policy": "detached_student_on_policy",
                "fp32_master_trainable_parameters": True,
            },
        }
        atomic_torch_save(
            payload,
            output_dir / "neodragon_rollout_bridge_latest.pt",
        )
        if archive:
            atomic_torch_save(
                payload,
                output_dir / f"neodragon_rollout_bridge_step{global_step:06d}.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        rank0_print(
            ctx,
            f"Saved rollout bridge step={global_step} latest=true archive={archive}",
        )
    barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue Exp1 bridge distillation through all 18 NeoDragon calls."
    )
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--manifest",
        default="data/openvid_neodragon_2s_latents/latent_manifest.csv",
    )
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--resume", default="auto")
    parser.add_argument(
        "--output-dir",
        default="output/neo_exp1_rollout_64k_to100k",
    )
    parser.add_argument("--target-step", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument("--rollout-ramp-steps", type=int, default=2000)
    parser.add_argument("--rollout-mse-weight", type=float, default=1.0)
    parser.add_argument("--rollout-cos-weight", type=float, default=0.1)
    parser.add_argument("--raw-token-weight", type=float, default=0.25)
    parser.add_argument("--normalized-token-weight", type=float, default=1.0)
    parser.add_argument("--cos-weight", type=float, default=0.5)
    parser.add_argument("--token-norm-weight", type=float, default=0.1)
    parser.add_argument("--pooled-weight", type=float, default=0.25)
    parser.add_argument("--pooled-cos-weight", type=float, default=0.2)
    parser.add_argument("--relational-weight", type=float, default=0.1)
    parser.add_argument("--generated-units", type=int, default=6)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--functional-batch-size", type=int, default=1)
    parser.add_argument(
        "--caption-variant-columns",
        default="caption_short,caption_medium,caption_long",
    )
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-latest-every", type=int, default=5000)
    parser.add_argument("--save-archive-every", type=int, default=10000)
    parser.add_argument("--archive-start-step", type=int, default=70000)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--parallel", choices=["none", "ddp"], default="ddp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_step < 1 or args.batch_size < 1 or args.functional_batch_size < 1:
        raise ValueError("target step and batch sizes must be positive")
    if args.generated_units * args.num_stages != 18:
        raise ValueError("This experiment must preserve NeoDragon hybrid's 6x3=18 calls.")
    if args.save_latest_every < 1 or args.save_archive_every < 1:
        raise ValueError("Checkpoint intervals must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ctx = setup_distributed()
    stage_log(ctx, "distributed process initialized")
    if args.parallel == "ddp" and not ctx.is_distributed:
        rank0_print(ctx, "WORLD_SIZE=1; using an unwrapped bridge for smoke testing.")
        args.parallel = "none"

    rank_seed = args.seed + ctx.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        # Each DDP process owns one CUDA device. Seeding every visible GPU from
        # every rank can create unnecessary cross-device CUDA contexts.
        torch.cuda.manual_seed(rank_seed)
    stage_log(ctx, f"random generators seeded with rank seed {rank_seed}")
    rollout_generator = torch.Generator(device=ctx.device)
    rollout_generator.manual_seed(rank_seed + 100_000)
    stage_log(ctx, "rollout CUDA generator ready")

    cfg = load_config(args.config)
    stage_log(ctx, f"configuration loaded from {args.config}")
    inference_dtype = dtype_from_name(args.dtype)
    if ctx.device.type == "cpu":
        inference_dtype = torch.float32
    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    stage_log(ctx, f"entering output-directory barrier for {output_dir}")
    barrier()
    stage_log(ctx, "output-directory barrier complete")

    latest_path = output_dir / "neodragon_rollout_bridge_latest.pt"
    resume_path: Path | None = None
    if args.resume == "auto" and latest_path.is_file():
        resume_path = latest_path
    elif args.resume not in {"", "none", "auto"}:
        resume_path = Path(args.resume)
    checkpoint_path = resume_path or Path(args.init_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing bridge checkpoint: {checkpoint_path}")

    stage_log(ctx, f"loading bridge from {checkpoint_path}")
    bridge, checkpoint = load_bridge(
        cfg,
        checkpoint_path,
        ctx.device,
        inference_dtype,
    )
    stage_log(ctx, "bridge loaded; trainable parameters use FP32 master weights")
    current_step = int(checkpoint.get("step", 0))
    initial_step = int(checkpoint.get("initial_step", current_step))
    if resume_path is None:
        initial_step = current_step
    if current_step >= args.target_step:
        rank0_print(
            ctx,
            f"Nothing to train: checkpoint_step={current_step} target_step={args.target_step}",
        )
        cleanup_distributed()
        return
    if current_step != 64000 and resume_path is None:
        raise ValueError(
            f"This experiment must initialize from Exp1-64k, got step={current_step}."
        )

    bridge_model: torch.nn.Module = bridge
    if args.parallel == "ddp":
        stage_log(ctx, "wrapping bridge with DistributedDataParallel")
        bridge_model = DDP(
            bridge,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
        )
        stage_log(ctx, "DistributedDataParallel ready")
    trainable = [parameter for parameter in bridge_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=0.0,
        foreach=False,
    )
    if resume_path is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    history = list(checkpoint.get("history", [])) if resume_path is not None else []

    caption_columns = [
        value.strip()
        for value in args.caption_variant_columns.split(",")
        if value.strip()
    ]
    caption_weights = [
        float(value.strip())
        for value in args.caption_variant_weights.split(",")
        if value.strip()
    ]
    stage_log(ctx, f"loading latent manifest {args.manifest}")
    dataset = VideoPromptDataset(
        args.manifest,
        max_samples=args.max_samples,
        caption_aug=True,
        caption_variant_columns=caption_columns,
        caption_variant_weights=caption_weights,
        caption_fallback_column=args.caption_fallback_column,
    )
    if not dataset.has_latents:
        raise ValueError("Full rollout distillation requires latent_path in the manifest.")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=args.seed,
        )
        if ctx.is_distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: collate_video_batch(
            batch,
            num_frames=cfg.data.frame_num,
            height=cfg.data.height,
            width=cfg.data.width,
            target_fps=24.0,
            latent_root=Path(args.manifest).expanduser().parent,
            use_latents=True,
        ),
    )
    stage_log(ctx, f"latent data loader ready with {len(dataset)} rows")
    start_epoch, start_batch = divmod(current_step, len(loader))
    batches = cycle_loader(
        loader,
        sampler,
        start_epoch=start_epoch,
        start_batch=start_batch,
    )

    stage_log(ctx, "loading frozen native NeoDragon text teacher")
    teacher_text, teacher_context_adapter, prompt_modifier = (
        load_neodragon_text_modules(cfg, ctx.device, inference_dtype)
    )
    stage_log(ctx, "native NeoDragon text teacher ready")
    stage_log(ctx, "loading frozen released hybrid NeoDragon DiT")
    frozen_dit, scheduler = load_neodragon_functional_modules(
        cfg,
        ctx.device,
        inference_dtype,
    )
    stage_log(ctx, "released hybrid NeoDragon DiT and scheduler ready")

    representation_weights = {
        "raw_token": args.raw_token_weight,
        "normalized_token": args.normalized_token_weight,
        "token_cosine": args.cos_weight,
        "token_norm": args.token_norm_weight,
        "pooled_mse": args.pooled_weight,
        "pooled_cosine": args.pooled_cos_weight,
        "relational": args.relational_weight,
    }
    rank0_print(
        ctx,
        "Full-rollout Exp1 continuation: "
        f"world_size={ctx.world_size} batch_per_gpu={args.batch_size} "
        f"global_batch={ctx.world_size * args.batch_size} "
        f"current_step={current_step} target_step={args.target_step} "
        f"rollout_calls={args.generated_units * args.num_stages} "
        f"trainable_params={sum(p.numel() for p in trainable):,} "
        f"master_dtype={next(iter(trainable)).dtype}",
    )
    rank0_print(
        ctx,
        f"Data rows={len(dataset)} loader_batches={len(loader)} "
        f"start_epoch={start_epoch} start_batch={start_batch}",
    )

    progress = tqdm(
        range(current_step + 1, args.target_step + 1),
        desc="Train Exp1 full NeoDragon rollout",
        disable=not ctx.is_main,
    )
    for global_step in progress:
        lr_scale = learning_rate_scale(
            global_step,
            initial_step=initial_step,
            target_step=args.target_step,
            warmup_steps=args.lr_warmup_steps,
            final_scale=args.lr_final_scale,
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        functional_scale = rollout_weight_scale(
            global_step,
            initial_step=initial_step,
            ramp_steps=args.rollout_ramp_steps,
        )

        batch = next(batches)
        prompts = [str(prompt) + prompt_modifier for prompt in batch["prompt"]]
        with torch.no_grad(), torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            teacher_tokens, teacher_mask, teacher_pooled = teacher_text(
                prompts,
                ctx.device,
            )
            teacher_tokens = teacher_context_adapter(teacher_tokens)

        with torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            student_tokens, student_mask, student_pooled = bridge_model(prompts)
            representation = bridge_representation_losses(
                student_tokens,
                teacher_tokens,
                teacher_mask,
                student_pooled,
                teacher_pooled,
            )
            representation_loss = weighted_loss_sum(
                representation,
                representation_weights,
            )

            effect_batch = min(args.functional_batch_size, len(prompts))
            latents = batch["latents"][:effect_batch].to(
                device=ctx.device,
                dtype=inference_dtype,
                non_blocking=True,
            )
            rollout = rollout_distillation_loss(
                dit=frozen_dit,
                scheduler=scheduler,
                anchor_latent=latents[:, :, :1],
                student_tokens=student_tokens[:effect_batch],
                student_mask=student_mask[:effect_batch],
                student_pooled=student_pooled[:effect_batch],
                teacher_tokens=teacher_tokens[:effect_batch],
                teacher_mask=teacher_mask[:effect_batch],
                teacher_pooled=teacher_pooled[:effect_batch],
                num_generated_units=args.generated_units,
                num_stages=args.num_stages,
                generator=rollout_generator,
            )
            functional_loss = (
                args.rollout_mse_weight * rollout.mse
                + args.rollout_cos_weight * rollout.cosine
            )
            loss = representation_loss + functional_scale * functional_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            args.clip_grad_norm,
        )
        optimizer.step()

        if global_step % args.log_every == 0 or global_step == current_step + 1:
            with torch.no_grad():
                pred_norm = masked_mean_norm(
                    student_tokens.float(),
                    teacher_mask,
                ).mean()
                target_norm = masked_mean_norm(
                    teacher_tokens.float(),
                    teacher_mask,
                ).mean()
                mask_accuracy = (
                    student_mask.bool() == teacher_mask.bool()
                ).float().mean()
            item = {
                "step": float(global_step),
                "loss": scalar_mean(loss.detach(), ctx),
                "representation_loss": scalar_mean(
                    representation_loss.detach(),
                    ctx,
                ),
                "raw_token_loss": scalar_mean(representation["raw_token"], ctx),
                "normalized_token_loss": scalar_mean(
                    representation["normalized_token"],
                    ctx,
                ),
                "cos_loss": scalar_mean(representation["token_cosine"], ctx),
                "norm_loss": scalar_mean(representation["token_norm"], ctx),
                "pooled_loss": scalar_mean(representation["pooled_mse"], ctx),
                "pooled_cos_loss": scalar_mean(
                    representation["pooled_cosine"],
                    ctx,
                ),
                "relational_loss": scalar_mean(
                    representation["relational"],
                    ctx,
                ),
                "rollout_mse": scalar_mean(rollout.mse.detach(), ctx),
                "rollout_cosine": scalar_mean(rollout.cosine.detach(), ctx),
                "rollout_stage0_mse": scalar_mean(
                    rollout.stage_mse[0].detach(),
                    ctx,
                ),
                "rollout_stage1_mse": scalar_mean(
                    rollout.stage_mse[1].detach(),
                    ctx,
                ),
                "rollout_stage2_mse": scalar_mean(
                    rollout.stage_mse[2].detach(),
                    ctx,
                ),
                "rollout_unit0_mse": scalar_mean(
                    rollout.unit_mse[0].detach(),
                    ctx,
                ),
                "rollout_unit5_mse": scalar_mean(
                    rollout.unit_mse[-1].detach(),
                    ctx,
                ),
                "rollout_scale": float(functional_scale),
                "rollout_calls": float(rollout.calls),
                "lr": float(args.lr * lr_scale),
                "grad_norm": scalar_mean(grad_norm.detach(), ctx),
                "pred_norm": scalar_mean(pred_norm.detach(), ctx),
                "target_norm": scalar_mean(target_norm.detach(), ctx),
                "mask_accuracy": scalar_mean(mask_accuracy.detach(), ctx),
                "world_size": float(ctx.world_size),
                "gpu_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(ctx.device) / (1024**3))
                    if ctx.device.type == "cuda"
                    else 0.0
                ),
            }
            if ctx.is_main:
                history.append(item)
                progress.set_postfix(
                    loss=f"{item['loss']:.4f}",
                    rollout=f"{item['rollout_mse']:.4f}",
                    cos=f"{item['rollout_cosine']:.4f}",
                    lr=f"{item['lr']:.2e}",
                )

        save_latest = (
            global_step % args.save_latest_every == 0
            or global_step == args.target_step
        )
        save_archive = bool(
            global_step >= args.archive_start_step
            and global_step % args.save_archive_every == 0
        )
        if save_latest:
            save_checkpoint(
                bridge_model=bridge_model,
                optimizer=optimizer,
                parallel=args.parallel,
                output_dir=output_dir,
                global_step=global_step,
                initial_step=initial_step,
                init_checkpoint=str(args.init_checkpoint),
                cfg=cfg,
                args=args,
                history=history,
                ctx=ctx,
                archive=save_archive,
            )

    rank0_print(
        ctx,
        f"Completed rollout bridge training at global step {args.target_step}.",
    )
    cleanup_distributed()


if __name__ == "__main__":
    main()
