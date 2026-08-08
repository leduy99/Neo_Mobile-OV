#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.neodragon_hybrid_recovery import (
    DiTCondition,
    StageUnitScaleEMA,
    clean_endpoint_for_position,
    predict_velocity,
    relative_endpoint_l2,
)
from new_mobile_ov.training.neodragon_objectives import bridge_representation_losses
from new_mobile_ov.training.neodragon_opsd import (
    adaptive_base_trust_loss,
    balanced_joint_position,
    history_relative_l2,
    normalized_velocity_mse,
    relative_velocity_l2,
    rollout_opsd_state_to_position,
    teacher_advantage_gate,
    velocity_cosine_distance,
    velocity_rms,
)
from tools.train_neodragon_dit_bridge import (
    VideoPromptDataset,
    collate_video_batch,
    load_bridge,
    load_neodragon_train_modules,
)
from tools.train_neodragon_hybrid_recovery import (
    all_reduce_mean,
    build_optimizer_groups,
    dtype_from_name,
    learning_rate_scale,
    promote_trainable_to_fp32,
)
from tools.train_neodragon_text_bridge import (
    cycle_loader,
    load_neodragon_functional_modules,
)


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_bridge_checkpoint(
    path: str | Path,
    *,
    expected_step: int,
    expected_sha256: str,
    ctx,
) -> tuple[int, str]:
    """Inspect the 1 GiB initialization checkpoint once, then broadcast metadata."""
    metadata: list[object | None] = [None]
    if ctx.is_main:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        step = int(payload.get("step", -1))
        del payload
        sha256 = file_sha256(path)
        metadata[0] = {"step": step, "sha256": sha256}
    if ctx.is_distributed:
        dist.broadcast_object_list(metadata, src=0)
    result = metadata[0]
    if not isinstance(result, dict):
        raise RuntimeError("Failed to broadcast Exp1 bridge metadata.")
    step = int(result["step"])
    sha256 = str(result["sha256"])
    if step != expected_step:
        raise ValueError(
            f"Expected Exp1 bridge step={expected_step}, got step={step} "
            f"from {path}."
        )
    if expected_sha256 and sha256 != expected_sha256:
        raise ValueError(
            "Exp1 bridge SHA256 mismatch: "
            f"expected={expected_sha256} actual={sha256}."
        )
    return step, sha256


def bridge_anchor_loss(
    *,
    student_tokens: torch.Tensor,
    student_mask: torch.Tensor,
    student_pooled: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_mask: torch.Tensor,
    teacher_pooled: torch.Tensor,
    normalized_token_weight: float,
    token_cos_weight: float,
    token_norm_weight: float,
    pooled_cos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            f"Bridge token shape mismatch: {tuple(student_tokens.shape)} vs "
            f"{tuple(teacher_tokens.shape)}."
        )
    if not torch.equal(student_mask.bool(), teacher_mask.bool()):
        raise ValueError("Student and fixed Exp1 bridge masks differ.")
    parts = bridge_representation_losses(
        student_tokens,
        teacher_tokens,
        teacher_mask,
        student_pooled,
        teacher_pooled,
    )
    loss = (
        float(normalized_token_weight) * parts["normalized_token"]
        + float(token_cos_weight) * parts["token_cosine"]
        + float(token_norm_weight) * parts["token_norm"]
        + float(pooled_cos_weight) * parts["pooled_cosine"]
    )
    return loss, parts


def bridge_lr_ramp(step: int, *, hold_steps: int, ramp_steps: int) -> float:
    if step <= hold_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(max((step - hold_steps) / float(ramp_steps), 0.0), 1.0)


def module_gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        raise RuntimeError("A jointly trained module received no gradients.")
    return torch.stack(norms).norm(2)


def save_checkpoint(
    *,
    student_model: torch.nn.Module,
    bridge_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    init_bridge_checkpoint: str,
    init_bridge_step: int,
    init_bridge_sha256: str,
    scale_ema: StageUnitScaleEMA,
    history: list[dict[str, object]],
    cfg,
    args: argparse.Namespace,
    archive: bool,
    save_optimizer: bool,
    ctx,
) -> None:
    if ctx.is_main:
        dit_state = {
            key: value.detach().cpu()
            for key, value in unwrap(student_model).state_dict().items()
        }
        bridge_state = {
            key: value.detach().cpu()
            for key, value in unwrap(bridge_model).state_dict().items()
        }
        model_payload = {
            "step": int(step),
            "dit": dit_state,
            "bridge": bridge_state,
            "init_bridge_checkpoint": init_bridge_checkpoint,
            "init_bridge_step": int(init_bridge_step),
            "init_bridge_sha256": init_bridge_sha256,
            "velocity_scale_ema": scale_ema.state_dict(),
            "history": history,
            "config": cfg,
            "args": vars(args),
            "objective": {
                "name": "opsd_neo_joint_v1",
                "student_policy": "released_hybrid_1-1-1_generated_history",
                "teacher": "frozen_released_hybrid_exp1_64k",
                "teacher_context": "gt_old_history_keep_latest_generated",
                "teacher_trajectory": "student_state",
                "teacher_quality_gate": "positive_clean_endpoint_advantage",
                "target": "velocity",
                "gt_output_target": False,
                "random_flow_matching": False,
                "trainable": "full_hybrid_dit_and_exp1_bridge",
                "truncation": "one_balanced_stage_unit_call_per_optimizer_step",
                "early_call_safeguard": "frozen_hybrid_velocity_preservation",
            },
        }
        latest_payload = dict(model_payload)
        if save_optimizer:
            latest_payload["optimizer"] = optimizer.state_dict()
        atomic_torch_save(
            latest_payload,
            output_dir / "neodragon_opsd_joint_latest.pt",
        )
        if archive:
            atomic_torch_save(
                model_payload,
                output_dir / f"neodragon_opsd_joint_step{step:06d}.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        rank0_print(
            ctx,
            f"Saved OPSD-Neo joint checkpoint step={step} archive={archive}",
        )
    barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fine-tune the Exp1 bridge and released NeoDragon Hybrid "
            "with privileged-history on-policy velocity distillation."
        )
    )
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--manifest",
        default="data/openvid_neodragon_2s_latents/latent_manifest.csv",
    )
    parser.add_argument("--bridge-ckpt", required=True)
    parser.add_argument("--bridge-expected-step", type=int, default=64000)
    parser.add_argument("--bridge-sha256", default="")
    parser.add_argument("--output-dir", default="output/neo_opsd_joint")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--min-loss-unit", type=int, default=2)
    parser.add_argument("--keep-recent-generated-units", type=int, default=1)
    parser.add_argument("--position-offset", type=int, default=0)

    parser.add_argument("--dit-middle-lr", type=float, default=2e-7)
    parser.add_argument("--dit-edge-lr", type=float, default=5e-8)
    parser.add_argument("--dit-io-lr", type=float, default=1e-7)
    parser.add_argument("--bridge-lr", type=float, default=2e-6)
    parser.add_argument("--edge-blocks", type=int, default=3)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-final-scale", type=float, default=0.2)
    parser.add_argument("--bridge-hold-steps", type=int, default=1000)
    parser.add_argument("--bridge-ramp-steps", type=int, default=2000)

    parser.add_argument("--opsd-weight", type=float, default=1.0)
    parser.add_argument("--velocity-cos-weight", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=0.25)
    parser.add_argument("--bridge-anchor-weight", type=float, default=0.02)
    parser.add_argument("--early-preserve-weight", type=float, default=0.10)
    parser.add_argument("--anchor-normalized-token-weight", type=float, default=1.0)
    parser.add_argument("--anchor-token-cos-weight", type=float, default=0.5)
    parser.add_argument("--anchor-token-norm-weight", type=float, default=0.1)
    parser.add_argument("--anchor-pooled-cos-weight", type=float, default=0.2)
    parser.add_argument("--trust-margin-scale", type=float, default=1.0)
    parser.add_argument("--trust-min-margin", type=float, default=0.01)
    parser.add_argument("--trust-max-margin", type=float, default=0.50)
    parser.add_argument("--teacher-gate-margin", type=float, default=0.0)
    parser.add_argument("--teacher-gate-ramp", type=float, default=0.05)
    parser.add_argument("--teacher-gate-check-step", type=int, default=120)
    parser.add_argument("--min-active-teacher-fraction", type=float, default=0.10)
    parser.add_argument("--normalizer-decay", type=float, default=0.99)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-latest-every", type=int, default=1000)
    parser.add_argument("--save-archive-every", type=int, default=5000)
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--caption-variant-columns",
        default="caption_short,caption_medium,caption_long",
    )
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive.")
    if args.num_units < 3 or args.num_stages < 1:
        raise ValueError("OPSD-Neo requires at least three units and one stage.")
    if not 2 <= args.min_loss_unit < args.num_units:
        raise ValueError(
            "min_loss_unit must be >= 2 so privileged history differs from "
            "generated history."
        )
    if not 1 <= args.keep_recent_generated_units < args.min_loss_unit + 1:
        raise ValueError(
            "keep_recent_generated_units must retain at least one unit while "
            "leaving older history available for replacement."
        )
    if args.bridge_expected_step < 0:
        raise ValueError("bridge_expected_step must be non-negative.")
    for name, value in vars(args).items():
        if (name.endswith("_weight") or name.endswith("_lr")) and value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if args.save_latest_every < 1 or args.save_archive_every < 1:
        raise ValueError("Checkpoint intervals must be positive.")
    if args.teacher_gate_ramp <= 0.0 or args.teacher_gate_check_step < 1:
        raise ValueError("Teacher gate ramp and check step must be positive.")
    if not 0.0 <= args.min_active_teacher_fraction <= 1.0:
        raise ValueError("min_active_teacher_fraction must be in [0, 1].")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ctx = setup_distributed()
    if not ctx.is_distributed:
        rank0_print(ctx, "WORLD_SIZE=1: running an unwrapped smoke test.")

    rank_seed = args.seed + ctx.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
    generator = torch.Generator(device=ctx.device).manual_seed(rank_seed + 100_000)

    cfg = load_config(args.config)
    inference_dtype = dtype_from_name(args.dtype)
    if ctx.device.type == "cpu":
        inference_dtype = torch.float32
    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    bridge_step, actual_bridge_sha256 = inspect_bridge_checkpoint(
        args.bridge_ckpt,
        expected_step=args.bridge_expected_step,
        expected_sha256=args.bridge_sha256,
        ctx=ctx,
    )

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
    dataset = VideoPromptDataset(
        args.manifest,
        max_samples=args.max_samples,
        caption_aug=True,
        caption_variant_columns=caption_columns,
        caption_variant_weights=caption_weights,
        caption_fallback_column=args.caption_fallback_column,
    )
    if not dataset.has_latents:
        raise ValueError("OPSD-Neo requires precomputed NeoDragon VAE latents.")
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

    rank0_print(ctx, "Loading trainable released Hybrid and Exp1-64K bridge.")
    student, _, scheduler, prompt_modifier = load_neodragon_train_modules(
        cfg,
        ctx.device,
        inference_dtype,
        load_vae=False,
    )
    bridge = load_bridge(
        cfg,
        args.bridge_ckpt,
        ctx.device,
        inference_dtype,
        trainable=True,
    )
    student.train().requires_grad_(True)
    student.gradient_checkpointing = bool(args.gradient_checkpointing)
    student.gradient_checkpointing_ratio = 0.0
    promote_trainable_to_fp32(student)
    promote_trainable_to_fp32(bridge)

    resume_path: Path | None = None
    latest_path = output_dir / "neodragon_opsd_joint_latest.pt"
    if args.resume == "auto" and latest_path.is_file():
        resume_path = latest_path
    elif args.resume not in {"", "none", "auto"}:
        resume_path = Path(args.resume)
    resume_payload: dict | None = None
    start_step = 0
    history: list[dict[str, object]] = []
    if resume_path is not None:
        resume_payload = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        student.load_state_dict(resume_payload["dit"], strict=True)
        bridge.load_state_dict(resume_payload["bridge"], strict=True)
        start_step = int(resume_payload["step"])
        history = list(resume_payload.get("history", []))
        saved_sha = str(resume_payload.get("init_bridge_sha256", ""))
        if saved_sha and saved_sha != actual_bridge_sha256:
            raise ValueError(
                "Refusing to resume with a different Exp1 bridge: "
                f"saved={saved_sha} current={actual_bridge_sha256}."
            )
        rank0_print(ctx, f"Resuming OPSD-Neo from {resume_path} at step={start_step}.")

    dit_groups, dit_counts = build_optimizer_groups(
        student,
        middle_lr=args.dit_middle_lr,
        edge_lr=args.dit_edge_lr,
        io_lr=args.dit_io_lr,
        edge_blocks=args.edge_blocks,
    )
    bridge_parameters = [
        parameter for parameter in bridge.parameters() if parameter.requires_grad
    ]
    dit_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer_groups = list(dit_groups) + [
        {
            "params": bridge_parameters,
            "lr": float(args.bridge_lr),
            "weight_decay": 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if resume_payload is not None and resume_payload.get("optimizer") is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    velocity_scale_ema = StageUnitScaleEMA(
        num_units=args.num_units,
        num_stages=args.num_stages,
        decay=args.normalizer_decay,
    )
    if resume_payload is not None and resume_payload.get("velocity_scale_ema") is not None:
        velocity_scale_ema.load_state_dict(resume_payload["velocity_scale_ema"])

    student_model: torch.nn.Module = student
    bridge_model: torch.nn.Module = bridge
    if ctx.is_distributed:
        student_model = DDP(
            student,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        bridge_model = DDP(
            bridge,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    rank0_print(ctx, "Loading frozen released-Hybrid and Exp1-64K teachers.")
    base_teacher, _ = load_neodragon_functional_modules(
        cfg,
        ctx.device,
        inference_dtype,
    )
    teacher_bridge = load_bridge(
        cfg,
        args.bridge_ckpt,
        ctx.device,
        inference_dtype,
        trainable=False,
    ).eval()
    base_teacher.eval().requires_grad_(False)
    teacher_bridge.eval().requires_grad_(False)

    start_epoch, start_batch = divmod(start_step, len(loader))
    batches = cycle_loader(
        loader,
        sampler,
        start_epoch=start_epoch,
        start_batch=start_batch,
    )
    trainable_dit = sum(parameter.numel() for parameter in student.parameters())
    trainable_bridge = sum(parameter.numel() for parameter in bridge_parameters)
    rank0_print(
        ctx,
        "OPSD-Neo joint ready: "
        f"world_size={ctx.world_size} batch_per_gpu={args.batch_size} "
        f"global_batch={ctx.world_size * args.batch_size} rows={len(dataset)} "
        f"dit_params={trainable_dit:,} bridge_params={trainable_bridge:,} "
        f"dit_groups={dit_counts} bridge_step={bridge_step} "
        f"positions=all_{args.num_units}x{args.num_stages} "
        f"opsd_units=[{args.min_loss_unit},{args.num_units - 1}] "
        f"data_epoch={start_epoch} data_batch={start_batch}",
    )
    if start_step >= args.steps:
        rank0_print(ctx, f"Nothing to train: step={start_step} target={args.steps}.")
        cleanup_distributed()
        return

    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        desc="Train OPSD-Neo joint bridge + Hybrid DiT",
        disable=not ctx.is_main,
    )
    active_teacher_sum = 0.0
    active_teacher_count = 0
    for step in progress:
        step_started = time.perf_counter()
        step_seed = args.seed + ctx.rank * 10_000_019 + step * 1_000_003
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(step_seed)
        generator.manual_seed(step_seed + 100_000)

        unit, stage = balanced_joint_position(
            step,
            num_units=args.num_units,
            num_stages=args.num_stages,
            offset=args.position_offset,
        )
        is_opsd_position = unit >= args.min_loss_unit
        lr_scale = learning_rate_scale(
            step,
            total_steps=args.steps,
            warmup_steps=args.lr_warmup_steps,
            final_scale=args.lr_final_scale,
        )
        bridge_scale = bridge_lr_ramp(
            step,
            hold_steps=args.bridge_hold_steps,
            ramp_steps=args.bridge_ramp_steps,
        )
        for index, (group, base_lr) in enumerate(zip(optimizer.param_groups, base_lrs)):
            group_scale = bridge_scale if index == len(optimizer.param_groups) - 1 else 1.0
            group["lr"] = base_lr * lr_scale * group_scale

        batch = next(batches)
        clean_latents = batch["latents"].to(
            device=ctx.device,
            dtype=inference_dtype,
            non_blocking=True,
        )
        expected_t = args.num_units + 1
        if clean_latents.shape[2] != expected_t:
            raise ValueError(
                f"OPSD-Neo expects anchor + {args.num_units} units (T={expected_t}), "
                f"got latent shape={tuple(clean_latents.shape)}."
            )
        prompts = [str(value) + prompt_modifier for value in batch["prompt"]]
        full_noise = torch.randn(
            clean_latents.shape,
            device=ctx.device,
            dtype=inference_dtype,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            student_tokens, student_mask, student_pooled = bridge_model(prompts)
        student_condition_detached = DiTCondition(
            tokens=student_tokens.detach(),
            mask=student_mask.detach(),
            pooled=student_pooled.detach(),
        )

        with torch.no_grad(), torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            teacher_tokens, teacher_mask, teacher_pooled = teacher_bridge(prompts)
            teacher_condition = DiTCondition(
                tokens=teacher_tokens,
                mask=teacher_mask,
                pooled=teacher_pooled,
            )
            state = rollout_opsd_state_to_position(
                actor=student_model,
                scheduler=scheduler,
                clean_latents=clean_latents,
                full_noise=full_noise,
                condition=student_condition_detached,
                target_unit=unit,
                target_stage=stage,
                keep_recent_generated=args.keep_recent_generated_units,
                num_stages=args.num_stages,
                generator=generator,
            )
            timestep = scheduler.get_stage_timesteps(
                1,
                stage,
                device=ctx.device,
            )[0]
            privileged_velocity = predict_velocity(
                dit=base_teacher,
                current=state.start,
                history=state.teacher_history,
                condition=teacher_condition,
                timestep=timestep,
            )
            base_velocity = predict_velocity(
                dit=base_teacher,
                current=state.start,
                history=state.student_history,
                condition=teacher_condition,
                timestep=timestep,
            )
            global_scale = all_reduce_mean(velocity_rms(privileged_velocity))
            normalizer = velocity_scale_ema.update(
                unit,
                stage,
                float(global_scale.cpu()),
            )
            context_gap = history_relative_l2(
                state.student_history,
                state.teacher_history,
            )
            sigmas = scheduler.get_stage_sigmas(1, stage, device=ctx.device)
            base_endpoint = scheduler.step(
                model_output=base_velocity,
                sigma=sigmas[0].to(dtype=base_velocity.dtype),
                sigma_next=sigmas[1].to(dtype=base_velocity.dtype),
                sample=state.start,
            ).prev_sample
            privileged_endpoint = scheduler.step(
                model_output=privileged_velocity,
                sigma=sigmas[0].to(dtype=privileged_velocity.dtype),
                sigma_next=sigmas[1].to(dtype=privileged_velocity.dtype),
                sample=state.start,
            ).prev_sample
            clean_endpoint = clean_endpoint_for_position(
                clean_latents,
                unit=unit,
                stage=stage,
                num_stages=args.num_stages,
            )
            base_gt_error = relative_endpoint_l2(
                base_endpoint,
                clean_endpoint,
                start=state.start,
            )
            privileged_gt_error = relative_endpoint_l2(
                privileged_endpoint,
                clean_endpoint,
                start=state.start,
            )
            if is_opsd_position:
                teacher_gate, teacher_relative_gain = teacher_advantage_gate(
                    base_gt_error,
                    privileged_gt_error,
                    margin=args.teacher_gate_margin,
                    ramp=args.teacher_gate_ramp,
                )
            else:
                teacher_gate = base_gt_error.new_zeros(())
                teacher_relative_gain = base_gt_error.new_zeros(())

        with torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            student_condition = DiTCondition(
                tokens=student_tokens,
                mask=student_mask,
                pooled=student_pooled,
            )
            student_velocity = predict_velocity(
                dit=student_model,
                current=state.start.detach(),
                history=tuple(value.detach() for value in state.student_history),
                condition=student_condition,
                timestep=timestep,
            )
            opsd_loss = normalized_velocity_mse(
                student_velocity,
                privileged_velocity.detach(),
                scale=normalizer,
            )
            velocity_cosine = velocity_cosine_distance(
                student_velocity,
                privileged_velocity.detach(),
            )
            trust_loss, student_base_gap, teacher_context_signal, trust_margin = (
                adaptive_base_trust_loss(
                    student_velocity,
                    base_velocity.detach(),
                    privileged_velocity.detach(),
                    scale=normalizer,
                    margin_scale=args.trust_margin_scale * teacher_gate,
                    minimum_margin=args.trust_min_margin,
                    maximum_margin=args.trust_max_margin,
                )
            )
            early_preserve_loss = normalized_velocity_mse(
                student_velocity,
                base_velocity.detach(),
                scale=normalizer,
            )
            anchor_loss, anchor_parts = bridge_anchor_loss(
                student_tokens=student_tokens,
                student_mask=student_mask,
                student_pooled=student_pooled,
                teacher_tokens=teacher_tokens.detach(),
                teacher_mask=teacher_mask.detach(),
                teacher_pooled=teacher_pooled.detach(),
                normalized_token_weight=args.anchor_normalized_token_weight,
                token_cos_weight=args.anchor_token_cos_weight,
                token_norm_weight=args.anchor_token_norm_weight,
                pooled_cos_weight=args.anchor_pooled_cos_weight,
            )
            loss = (
                teacher_gate * args.opsd_weight * opsd_loss
                + teacher_gate * args.velocity_cos_weight * velocity_cosine
                + args.trust_weight * trust_loss
                + args.bridge_anchor_weight * anchor_loss
                + (
                    args.early_preserve_weight * early_preserve_loss
                    if not is_opsd_position
                    else 0.0
                )
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite OPSD-Neo loss at step={step}: {float(loss.detach())}."
            )
        loss.backward()
        if step == start_step + 1:
            if not any(parameter.grad is not None for parameter in dit_parameters):
                raise RuntimeError("NeoDragon DiT received no gradients.")
            if not any(parameter.grad is not None for parameter in bridge_parameters):
                raise RuntimeError("Mobile-OV bridge received no gradients.")
        bridge_grad_norm = module_gradient_norm(bridge_parameters)
        trainable_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            args.clip_grad_norm,
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at step={step}: {float(grad_norm)}."
            )
        optimizer.step()

        if is_opsd_position:
            globally_active = scalar_mean((teacher_gate > 0).float(), ctx)
            active_teacher_sum += globally_active
            active_teacher_count += 1
        if step == args.teacher_gate_check_step and start_step < args.teacher_gate_check_step:
            active_fraction = active_teacher_sum / max(active_teacher_count, 1)
            if active_fraction < args.min_active_teacher_fraction:
                if ctx.is_main:
                    failure = {
                        "step": int(step),
                        "active_teacher_fraction": float(active_fraction),
                        "minimum_required": float(args.min_active_teacher_fraction),
                        "teacher_gate_margin": float(args.teacher_gate_margin),
                        "teacher_gate_ramp": float(args.teacher_gate_ramp),
                        "decision": "abort_before_scaling",
                    }
                    (output_dir / "opsd_teacher_gate_failure.json").write_text(
                        json.dumps(failure, indent=2),
                        encoding="utf-8",
                    )
                    (output_dir / "history.json").write_text(
                        json.dumps(history, indent=2),
                        encoding="utf-8",
                    )
                raise RuntimeError(
                    "Privileged-history teacher failed the safety check: "
                    f"active_fraction={active_fraction:.4f} < "
                    f"minimum={args.min_active_teacher_fraction:.4f}. "
                    "Do not continue OPSD-Neo until teacher context is validated."
                )

        should_log = step % args.log_every == 0 or step == start_step + 1
        if should_log:
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            with torch.no_grad():
                student_target_relative = relative_velocity_l2(
                    student_velocity,
                    privileged_velocity,
                    scale=normalizer,
                )
                teacher_gt_gain = base_gt_error - privileged_gt_error

            item: dict[str, object] = {
                "step": int(step),
                "unit": int(unit),
                "stage": int(stage),
                "loss": scalar_mean(loss.detach(), ctx),
                "opsd_velocity_mse": scalar_mean(opsd_loss.detach(), ctx),
                "opsd_velocity_cosine": scalar_mean(velocity_cosine.detach(), ctx),
                "early_preserve_loss": scalar_mean(
                    early_preserve_loss.detach(),
                    ctx,
                ),
                "training_mode": "opsd" if is_opsd_position else "early_preserve",
                "trust_loss": scalar_mean(trust_loss.detach(), ctx),
                "student_base_gap": scalar_mean(student_base_gap.detach(), ctx),
                "teacher_context_signal": scalar_mean(teacher_context_signal.detach(), ctx),
                "trust_margin": scalar_mean(trust_margin.detach(), ctx),
                "student_target_relative_l2": scalar_mean(
                    student_target_relative.detach(),
                    ctx,
                ),
                "history_relative_l2": scalar_mean(context_gap.detach(), ctx),
                "bridge_anchor": scalar_mean(anchor_loss.detach(), ctx),
                "bridge_token_cosine": scalar_mean(
                    anchor_parts["token_cosine"].detach(),
                    ctx,
                ),
                "bridge_pooled_cosine": scalar_mean(
                    anchor_parts["pooled_cosine"].detach(),
                    ctx,
                ),
                "base_gt_endpoint_error_diagnostic": scalar_mean(
                    base_gt_error.detach(),
                    ctx,
                ),
                "privileged_gt_endpoint_error_diagnostic": scalar_mean(
                    privileged_gt_error.detach(),
                    ctx,
                ),
                "teacher_gt_gain_diagnostic": scalar_mean(
                    teacher_gt_gain.detach(),
                    ctx,
                ),
                "teacher_relative_gain": scalar_mean(
                    teacher_relative_gain.detach(),
                    ctx,
                ),
                "teacher_gate": scalar_mean(teacher_gate.detach(), ctx),
                "active_teacher_fraction_running": (
                    active_teacher_sum / max(active_teacher_count, 1)
                ),
                "velocity_normalizer": float(normalizer),
                "lr_scale": float(lr_scale),
                "bridge_lr_scale": float(bridge_scale),
                "dit_middle_lr": float(args.dit_middle_lr * lr_scale),
                "bridge_lr": float(args.bridge_lr * lr_scale * bridge_scale),
                "grad_norm": scalar_mean(grad_norm.detach(), ctx),
                "bridge_grad_norm": scalar_mean(bridge_grad_norm.detach(), ctx),
                "step_seconds": float(time.perf_counter() - step_started),
                "gpu_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(ctx.device) / (1024**3))
                    if ctx.device.type == "cuda"
                    else 0.0
                ),
            }
            if ctx.is_main:
                history.append(item)
                progress.set_postfix(
                    pos=f"{unit}/{stage}",
                    mode=item["training_mode"],
                    loss=f"{item['loss']:.4f}",
                    opsd=f"{item['opsd_velocity_mse']:.4f}",
                    signal=f"{item['teacher_context_signal']:.3f}",
                    trust=f"{item['trust_loss']:.3f}",
                )

        is_final = step == args.steps
        save_latest = (
            step % args.save_latest_every == 0
            or (is_final and args.save_final)
        )
        save_archive = (
            step % args.save_archive_every == 0
            or (is_final and args.save_final)
        )
        if save_latest:
            save_checkpoint(
                student_model=student_model,
                bridge_model=bridge_model,
                optimizer=optimizer,
                output_dir=output_dir,
                step=step,
                init_bridge_checkpoint=args.bridge_ckpt,
                init_bridge_step=bridge_step,
                init_bridge_sha256=actual_bridge_sha256,
                scale_ema=velocity_scale_ema,
                history=history,
                cfg=cfg,
                args=args,
                archive=save_archive,
                save_optimizer=args.save_optimizer,
                ctx=ctx,
            )

    if ctx.is_main:
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
    barrier()
    rank0_print(ctx, f"Completed OPSD-Neo joint training at step={args.steps}.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
