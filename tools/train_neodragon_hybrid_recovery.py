#!/usr/bin/env python
# ruff: noqa: E402
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
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
    balanced_position,
    clean_endpoint_for_position,
    corrupt_history,
    endpoint_cosine_distance,
    hybrid_trust_region_loss,
    normalized_charbonnier,
    relative_endpoint_l2,
    rollout_state_to_position,
    run_stage_endpoint,
    sample_curriculum_mode,
    teacher_forced_state_to_position,
    transition_rms,
)
from tools.train_neodragon_dit_bridge import (
    VideoPromptDataset,
    collate_video_batch,
    cycle_loader,
    load_bridge,
    load_neodragon_train_modules,
)
from tools.train_neodragon_text_bridge import load_neodragon_functional_modules


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def promote_trainable_to_fp32(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def build_optimizer_groups(
    dit: torch.nn.Module,
    *,
    middle_lr: float,
    edge_lr: float,
    io_lr: float,
    edge_blocks: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("NeoDragon DiT does not expose transformer_blocks.")
    num_blocks = len(blocks)
    if not 0 <= edge_blocks * 2 <= num_blocks:
        raise ValueError(
            f"edge_blocks={edge_blocks} is invalid for {num_blocks} DiT blocks."
        )

    groups: dict[str, dict[str, object]] = {
        "edge": {"params": [], "lr": float(edge_lr), "weight_decay": 0.0},
        "middle": {"params": [], "lr": float(middle_lr), "weight_decay": 0.0},
        "io": {"params": [], "lr": float(io_lr), "weight_decay": 0.0},
    }
    for name, parameter in dit.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = "io"
        if name.startswith("transformer_blocks."):
            block_index = int(name.split(".", 2)[1])
            if block_index < edge_blocks or block_index >= num_blocks - edge_blocks:
                group_name = "edge"
            else:
                group_name = "middle"
        groups[group_name]["params"].append(parameter)

    counts = {
        name: sum(parameter.numel() for parameter in group["params"])
        for name, group in groups.items()
    }
    return [group for group in groups.values() if group["params"]], counts


def learning_rate_scale(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    final_scale: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return max(step / float(warmup_steps), 1e-3)
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_scale) + (1.0 - float(final_scale)) * cosine


def load_monolithic_teacher(
    cfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import (
        MULTISTEP_CONTEXT_ADAPTER_ID,
        MULTISTEP_DIT_ID,
    )
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_PROMPT_MODIFIER,
    )

    text_bundle = TextEncoderBundle.from_pretrained(
        local_model_path,
        torch_dtype=dtype,
    ).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    dit = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    for module in [text_bundle, context_adapter, dit]:
        module.requires_grad_(False)
    return (
        text_bundle,
        context_adapter,
        dit,
        DEFAULT_PROMPT_MODIFIER,
        DEFAULT_NEGATIVE_PROMPT,
    )


def native_condition(
    *,
    text_bundle,
    context_adapter,
    prompts: list[str],
    negative_prompt: str,
    device: torch.device,
    guidance_scale: float,
) -> DiTCondition:
    tokens, mask, pooled = text_bundle(prompts, device)
    tokens = context_adapter(tokens)
    negatives = [negative_prompt] * len(prompts)
    negative_tokens, negative_mask, negative_pooled = text_bundle(negatives, device)
    negative_tokens = context_adapter(negative_tokens)
    return DiTCondition(
        tokens=tokens,
        mask=mask,
        pooled=pooled,
        negative_tokens=negative_tokens,
        negative_mask=negative_mask,
        negative_pooled=negative_pooled,
        guidance_scale=float(guidance_scale),
    )


def bridge_condition(
    bridge: MobileOVNeodragonTextBridge,
    prompts: list[str],
) -> DiTCondition:
    tokens, mask, pooled = bridge(prompts)
    return DiTCondition(tokens=tokens, mask=mask, pooled=pooled)


def save_checkpoint(
    *,
    student_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    bridge_checkpoint: str,
    scale_ema: StageUnitScaleEMA,
    history: list[dict[str, object]],
    cfg,
    args: argparse.Namespace,
    archive: bool,
    save_optimizer: bool,
    ctx,
) -> None:
    if ctx.is_main:
        student_state = {
            key: value.detach().cpu()
            for key, value in unwrap(student_model).state_dict().items()
        }
        model_payload = {
            "step": int(step),
            "dit": student_state,
            "bridge_ckpt": bridge_checkpoint,
            "scale_ema": scale_ema.state_dict(),
            "history": history,
            "config": cfg,
            "args": vars(args),
            "objective": {
                "name": "hybrid_transition_recovery_v1",
                "student": "released_hybrid_full_weight",
                "monolithic_teacher": "released_multistep_native_cfg",
                "hybrid_teacher": "released_hybrid_exp1_condition",
                "schedule": "1-1-1",
                "positions": "6_units_x_3_stages_balanced",
                "map_target": "stage_endpoint",
                "on_policy": True,
                "hybrid_trust_region": True,
                "full_dmd_fake_model": False,
                "midpoint_self_consistency": False,
            },
        }
        latest_payload = dict(model_payload)
        if save_optimizer:
            latest_payload["optimizer"] = optimizer.state_dict()
        atomic_torch_save(
            latest_payload,
            output_dir / "neodragon_exp6_latest.pt",
        )
        if archive:
            atomic_torch_save(
                model_payload,
                output_dir / f"neodragon_exp6_step{step:06d}.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        rank0_print(
            ctx,
            f"Saved Exp6 checkpoint step={step} latest=true archive={archive}",
        )
    barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover NeoDragon Hybrid's 18 one-step transport maps."
    )
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument(
        "--manifest",
        default="data/openvid_neodragon_2s_latents/latent_manifest.csv",
    )
    parser.add_argument("--bridge-ckpt", required=True)
    parser.add_argument("--output-dir", default="output/neo_exp6_hybrid_recovery")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--parity-steps", type=int, default=500)
    parser.add_argument("--map-end-step", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--middle-lr", type=float, default=1e-6)
    parser.add_argument("--edge-lr", type=float, default=2.5e-7)
    parser.add_argument("--io-lr", type=float, default=5e-7)
    parser.add_argument("--edge-blocks", type=int, default=3)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument("--map-weight", type=float, default=1.0)
    parser.add_argument("--map-cos-weight", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=0.15)
    parser.add_argument("--real-endpoint-weight", type=float, default=0.05)
    parser.add_argument("--trust-margin-scale", type=float, default=1.0)
    parser.add_argument("--trust-min-margin", type=float, default=0.02)
    parser.add_argument("--trust-max-margin", type=float, default=0.50)
    parser.add_argument("--normalizer-decay", type=float, default=0.99)
    parser.add_argument("--history-noise-max", type=float, default=1 / 3)
    parser.add_argument("--monolithic-steps", type=int, default=10)
    parser.add_argument("--monolithic-guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help="Diagnostic-only cyclic offset for the balanced 6x3 transition schedule.",
    )
    parser.add_argument(
        "--mode-override",
        choices=[
            "hybrid_parity",
            "teacher_map",
            "hybrid_replay",
            "student_replay",
            "noisy_history",
            "real_endpoint",
        ],
        default=None,
        help="Diagnostic-only override for the curriculum mode.",
    )
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
        help="Keep Adam state in latest; archives are model-only.",
    )
    parser.add_argument("--caption-variant-columns", default="caption_short,caption_medium,caption_long")
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive.")
    if not 0 <= args.parity_steps <= args.map_end_step <= args.steps:
        raise ValueError("Require 0 <= parity_steps <= map_end_step <= steps.")
    if args.num_units * args.num_stages != 18:
        raise ValueError("NeoDragon Hybrid recovery must preserve 6x3=18 calls.")
    if args.monolithic_steps < 2:
        raise ValueError("Monolithic teacher must use multiple denoising steps.")
    for name in [
        "middle_lr",
        "edge_lr",
        "io_lr",
        "map_weight",
        "map_cos_weight",
        "trust_weight",
        "real_endpoint_weight",
    ]:
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not 0.0 <= args.history_noise_max <= 1 / 3 + 1e-8:
        raise ValueError("history_noise_max must be within [0, 1/3].")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ctx = setup_distributed()
    if not ctx.is_distributed:
        rank0_print(ctx, "WORLD_SIZE=1: running an unwrapped single-GPU smoke test.")

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
        raise ValueError("Exp6 requires a manifest with precomputed latent_path.")
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
    batches = cycle_loader(loader, sampler)

    rank0_print(ctx, "Loading released Hybrid Student.")
    student, _, scheduler, _ = load_neodragon_train_modules(
        cfg,
        ctx.device,
        inference_dtype,
        load_vae=False,
    )
    student.requires_grad_(True)
    student.gradient_checkpointing = bool(args.gradient_checkpointing)
    student.gradient_checkpointing_ratio = 0.0
    promote_trainable_to_fp32(student)

    resume_path: Path | None = None
    latest_path = output_dir / "neodragon_exp6_latest.pt"
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
        start_step = int(resume_payload["step"])
        history = list(resume_payload.get("history", []))
        rank0_print(ctx, f"Resuming Exp6 from {resume_path} at step={start_step}.")

    optimizer_groups, parameter_counts = build_optimizer_groups(
        student,
        middle_lr=args.middle_lr,
        edge_lr=args.edge_lr,
        io_lr=args.io_lr,
        edge_blocks=args.edge_blocks,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])

    scale_ema = StageUnitScaleEMA(
        num_units=args.num_units,
        num_stages=args.num_stages,
        decay=args.normalizer_decay,
    )
    if resume_payload is not None and "scale_ema" in resume_payload:
        scale_ema.load_state_dict(resume_payload["scale_ema"])

    student_model: torch.nn.Module = student
    if ctx.is_distributed:
        student_model = DDP(
            student,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    rank0_print(ctx, "Loading released frozen Hybrid trust teacher.")
    hybrid_teacher, _ = load_neodragon_functional_modules(
        cfg,
        ctx.device,
        inference_dtype,
    )
    rank0_print(ctx, "Loading frozen Monolithic teacher and native conditioner.")
    (
        native_text,
        native_adapter,
        monolithic_teacher,
        prompt_modifier,
        negative_prompt,
    ) = load_monolithic_teacher(
        cfg,
        device=ctx.device,
        dtype=inference_dtype,
    )
    bridge = load_bridge(
        cfg,
        args.bridge_ckpt,
        ctx.device,
        inference_dtype,
        trainable=False,
    ).eval()

    rank0_print(
        ctx,
        "Exp6 Hybrid recovery ready: "
        f"world_size={ctx.world_size} batch_per_gpu={args.batch_size} "
        f"global_batch={ctx.world_size * args.batch_size} rows={len(dataset)} "
        f"student_params={sum(p.numel() for p in student.parameters()):,} "
        f"optimizer_groups={parameter_counts} "
        f"bridge={args.bridge_ckpt}",
    )
    if start_step >= args.steps:
        rank0_print(ctx, f"Nothing to train: step={start_step} target={args.steps}.")
        cleanup_distributed()
        return

    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        desc="Train NeoDragon Exp6 Hybrid recovery",
        disable=not ctx.is_main,
    )
    for step in progress:
        unit, stage = balanced_position(
            step,
            num_units=args.num_units,
            num_stages=args.num_stages,
            offset=args.position_offset,
        )
        mode = args.mode_override or sample_curriculum_mode(
            step,
            seed=args.seed,
            parity_steps=args.parity_steps,
            map_end_step=args.map_end_step,
        )
        lr_scale = learning_rate_scale(
            step,
            total_steps=args.steps,
            warmup_steps=args.lr_warmup_steps,
            final_scale=args.lr_final_scale,
        )
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * lr_scale

        batch = next(batches)
        clean_latents = batch["latents"].to(
            device=ctx.device,
            dtype=inference_dtype,
            non_blocking=True,
        )
        if clean_latents.shape[2] != args.num_units + 1:
            raise ValueError(
                f"Exp6 expects anchor + {args.num_units} units, got "
                f"latent shape={tuple(clean_latents.shape)}."
            )
        prompts = [str(value) + prompt_modifier for value in batch["prompt"]]
        full_noise = torch.randn(
            clean_latents.shape,
            device=ctx.device,
            dtype=inference_dtype,
            generator=generator,
        )

        with torch.no_grad(), torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            mobile_condition = bridge_condition(bridge, prompts)
            teacher_condition = native_condition(
                text_bundle=native_text,
                context_adapter=native_adapter,
                prompts=prompts,
                negative_prompt=negative_prompt,
                device=ctx.device,
                guidance_scale=args.monolithic_guidance_scale,
            )

            if mode in {"hybrid_parity", "hybrid_replay"}:
                actor = hybrid_teacher
                actor_condition = mobile_condition
                actor_steps = 1
            elif mode in {"student_replay", "noisy_history"}:
                actor = student_model
                actor_condition = mobile_condition
                actor_steps = 1
            else:
                actor = monolithic_teacher
                actor_condition = teacher_condition
                actor_steps = args.monolithic_steps

            if mode == "real_endpoint":
                state = teacher_forced_state_to_position(
                    actor=actor,
                    scheduler=scheduler,
                    clean_latents=clean_latents,
                    full_noise=full_noise,
                    condition=actor_condition,
                    target_unit=unit,
                    target_stage=stage,
                    actor_steps=actor_steps,
                    num_stages=args.num_stages,
                    generator=generator,
                )
            else:
                state = rollout_state_to_position(
                    actor=actor,
                    scheduler=scheduler,
                    anchor=clean_latents[:, :, :1],
                    full_noise=full_noise,
                    condition=actor_condition,
                    target_unit=unit,
                    target_stage=stage,
                    actor_steps=actor_steps,
                    num_stages=args.num_stages,
                    generator=generator,
                )
            target_history = state.history
            corruption_strength = 0.0
            if mode == "noisy_history":
                corruption_strength = random.random() * args.history_noise_max
                target_history = corrupt_history(
                    target_history,
                    strength=corruption_strength,
                    generator=generator,
                )

            hybrid_endpoint, _ = run_stage_endpoint(
                dit=hybrid_teacher,
                scheduler=scheduler,
                current=state.start,
                history=target_history,
                condition=mobile_condition,
                stage=stage,
                num_steps=1,
            )
            monolithic_endpoint: torch.Tensor | None = None
            if mode != "hybrid_parity":
                monolithic_endpoint, _ = run_stage_endpoint(
                    dit=monolithic_teacher,
                    scheduler=scheduler,
                    current=state.start,
                    history=target_history,
                    condition=teacher_condition,
                    stage=stage,
                    num_steps=args.monolithic_steps,
                )

            if mode == "hybrid_parity":
                target_endpoint = hybrid_endpoint
            elif mode == "real_endpoint":
                target_endpoint = clean_endpoint_for_position(
                    clean_latents,
                    unit=unit,
                    stage=stage,
                    num_stages=args.num_stages,
                )
            else:
                if monolithic_endpoint is None:
                    raise RuntimeError("Monolithic endpoint target was not computed.")
                target_endpoint = monolithic_endpoint

            local_scale = transition_rms(state.start, target_endpoint)
            global_scale = all_reduce_mean(local_scale)
            normalizer = scale_ema.update(
                unit,
                stage,
                float(global_scale.cpu()),
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=ctx.device.type,
            dtype=inference_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            student_endpoint, _ = run_stage_endpoint(
                dit=student_model,
                scheduler=scheduler,
                current=state.start.detach(),
                history=tuple(value.detach() for value in target_history),
                condition=mobile_condition,
                stage=stage,
                num_steps=1,
            )
            map_loss = normalized_charbonnier(
                student_endpoint,
                target_endpoint.detach(),
                scale=normalizer,
            )
            map_cosine = endpoint_cosine_distance(
                student_endpoint,
                target_endpoint.detach(),
            )

            trust_loss = student_endpoint.new_zeros(())
            student_hybrid_gap = student_endpoint.new_zeros(())
            trust_margin = student_endpoint.new_zeros(())
            if monolithic_endpoint is not None:
                trust_loss, student_hybrid_gap, trust_margin = (
                    hybrid_trust_region_loss(
                        student_endpoint,
                        hybrid_endpoint.detach(),
                        monolithic_endpoint.detach(),
                        start=state.start.detach(),
                        margin_scale=args.trust_margin_scale,
                        minimum_margin=args.trust_min_margin,
                        maximum_margin=args.trust_max_margin,
                    )
                )

            effective_map_weight = (
                args.real_endpoint_weight
                if mode == "real_endpoint"
                else args.map_weight
            )
            loss = (
                effective_map_weight * map_loss
                + args.map_cos_weight * map_cosine
                + args.trust_weight * trust_loss
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student_model.parameters() if parameter.requires_grad],
            args.clip_grad_norm,
        )
        optimizer.step()

        if step % args.log_every == 0 or step == start_step + 1:
            with torch.no_grad():
                target_relative_l2 = relative_endpoint_l2(
                    student_endpoint,
                    target_endpoint,
                    start=state.start,
                )
                hybrid_target_l2 = relative_endpoint_l2(
                    hybrid_endpoint,
                    target_endpoint,
                    start=state.start,
                )
            item: dict[str, object] = {
                "step": step,
                "mode": mode,
                "unit": unit,
                "stage": stage,
                "loss": scalar_mean(loss.detach(), ctx),
                "map_loss": scalar_mean(map_loss.detach(), ctx),
                "map_cosine": scalar_mean(map_cosine.detach(), ctx),
                "trust_loss": scalar_mean(trust_loss.detach(), ctx),
                "student_hybrid_gap": scalar_mean(student_hybrid_gap.detach(), ctx),
                "trust_margin": scalar_mean(trust_margin.detach(), ctx),
                "student_target_relative_l2": scalar_mean(target_relative_l2.detach(), ctx),
                "hybrid_target_relative_l2": scalar_mean(hybrid_target_l2.detach(), ctx),
                "transition_normalizer": float(normalizer),
                "history_corruption": float(corruption_strength),
                "lr_scale": float(lr_scale),
                "middle_lr": float(args.middle_lr * lr_scale),
                "grad_norm": scalar_mean(grad_norm.detach(), ctx),
                "gpu_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(ctx.device) / (1024**3))
                    if ctx.device.type == "cuda"
                    else 0.0
                ),
            }
            if ctx.is_main:
                history.append(item)
                progress.set_postfix(
                    mode=mode,
                    pos=f"{unit}/{stage}",
                    loss=f"{item['loss']:.4f}",
                    rel=f"{item['student_target_relative_l2']:.3f}",
                    trust=f"{item['trust_loss']:.3f}",
                )

        save_latest = step % args.save_latest_every == 0 or step == args.steps
        save_archive = step % args.save_archive_every == 0 or step == args.steps
        if save_latest:
            save_checkpoint(
                student_model=student_model,
                optimizer=optimizer,
                output_dir=output_dir,
                step=step,
                bridge_checkpoint=args.bridge_ckpt,
                scale_ema=scale_ema,
                history=history,
                cfg=cfg,
                args=args,
                archive=save_archive,
                save_optimizer=args.save_optimizer,
                ctx=ctx,
            )

    rank0_print(ctx, f"Completed Exp6 Hybrid recovery at step={args.steps}.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
