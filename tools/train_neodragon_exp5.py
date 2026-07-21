#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    full_optimizer_state_dict,
    full_state_dict,
    load_full_optimizer_state_dict,
    rank0_print,
    setup_distributed,
)
from new_mobile_ov.training.exp5_schedule import Exp5Schedule
from new_mobile_ov.training.neodragon_objectives import (
    bridge_representation_losses,
    flat_cosine_distance,
    relational_cosine,
    weighted_loss_sum,
)
from tools.train_neodragon_dit_bridge import (
    VideoPromptDataset,
    _condition_offdiag_cosine,
    _gather_detached,
    _gather_with_gradient,
    _shift_condition_across_global_batch,
    collate_video_batch,
    dtype_from_name,
    freeze_dit_for_last_n_blocks,
    load_bridge,
    load_neodragon_teacher_modules,
    load_neodragon_train_modules,
    wrap_bridge,
    wrap_dit,
)


class OffsetDistributedSampler(DistributedSampler):
    """Resume an epoch without decoding or loading already-consumed samples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_offset = 0

    def set_start_offset(self, sample_offset: int) -> None:
        if not 0 <= sample_offset <= self.num_samples:
            raise ValueError(f"sample_offset must be in [0, {self.num_samples}], got {sample_offset}")
        self.start_offset = int(sample_offset)

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_offset :])

    def __len__(self) -> int:
        return self.num_samples - self.start_offset


class MetricWindow:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, value: torch.Tensor | float, *, active: bool = True) -> None:
        if not active:
            return
        scalar = float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
        self.sums[name] += scalar
        self.counts[name] += 1

    def distributed_means(self, device: torch.device) -> dict[str, float]:
        names = sorted(self.sums)
        if not names:
            return {}
        packed = torch.tensor(
            [[self.sums[name], float(self.counts[name])] for name in names],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        means = {
            name: float(packed[index, 0].item() / max(packed[index, 1].item(), 1.0))
            for index, name in enumerate(names)
        }
        self.sums.clear()
        self.counts.clear()
        return means


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_resume_path(value: str, output_dir: Path) -> Path | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "false", "0"}:
        return None
    if normalized == "auto":
        candidate = output_dir / "neodragon_exp5_latest.pt"
        return candidate if candidate.is_file() else None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def validate_resume_compatibility(
    payload: dict[str, Any],
    args: argparse.Namespace,
    schedule: Exp5Schedule,
    world_size: int,
) -> None:
    """Reject accidental resumes with a different training experiment."""
    saved_schedule = payload.get("schedule")
    if saved_schedule != schedule.as_dict():
        raise ValueError(
            "Resume schedule differs from this run. Keep Exp5 phase boundaries and total steps "
            f"unchanged, or use a new --output-dir. saved={saved_schedule} current={schedule.as_dict()}"
        )

    saved_args = payload.get("args") or {}
    immutable_args = [
        "config",
        "manifest",
        "batch_size",
        "parallel",
        "train_last_n_blocks",
        "dit_lr",
        "bridge_lr",
        "distill_weight",
        "distill_cos_weight",
        "preservation_weight",
        "preservation_cos_weight",
        "preservation_every",
        "bridge_repr_weight",
        "bridge_raw_token_weight",
        "bridge_normalized_token_weight",
        "bridge_cos_weight",
        "bridge_token_norm_weight",
        "bridge_pooled_weight",
        "bridge_pooled_cos_weight",
        "bridge_relational_weight",
        "bridge_functional_weight",
        "bridge_functional_cos_weight",
        "bridge_functional_every",
        "bridge_functional_batch_size",
        "caption_variant_columns",
        "caption_variant_weights",
        "caption_fallback_column",
        "seed",
    ]
    changed = {
        name: {"saved": saved_args.get(name), "current": getattr(args, name)}
        for name in immutable_args
        if saved_args.get(name) != getattr(args, name)
    }
    saved_world_size = int((payload.get("parallel") or {}).get("world_size", world_size))
    if saved_world_size != world_size:
        changed["world_size"] = {"saved": saved_world_size, "current": world_size}
    if changed:
        raise ValueError(
            "Resume configuration differs from the saved Exp5 run. Use identical settings or a "
            f"new --output-dir. Differences: {changed}"
        )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state(sample_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "sample_generator": sample_generator.get_state(),
    }


def gather_rng_states(local_state: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


def restore_rng_state(state: dict[str, Any], sample_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])
    sample_generator.set_state(state["sample_generator"])


def iter_batches_from_step(
    loader: DataLoader,
    sampler: OffsetDistributedSampler,
    *,
    completed_steps: int,
    full_steps_per_epoch: int,
    batch_size: int,
):
    epoch = completed_steps // full_steps_per_epoch
    step_offset = completed_steps % full_steps_per_epoch
    first_epoch = True
    while True:
        sampler.set_epoch(epoch)
        sampler.set_start_offset(step_offset * batch_size if first_epoch else 0)
        for batch in loader:
            yield batch
        first_epoch = False
        step_offset = 0
        epoch += 1


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def clip_grad_norm(model: torch.nn.Module, max_norm: float, parallel: str) -> torch.Tensor:
    if max_norm <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    if parallel == "fsdp" and hasattr(model, "clip_grad_norm_"):
        return model.clip_grad_norm_(max_norm)
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def model_payload(
    *,
    step: int,
    phase: str,
    dit_state: dict[str, torch.Tensor],
    bridge_state: dict[str, torch.Tensor],
    bridge_ckpt: str,
    cfg: Any,
    args: argparse.Namespace,
    schedule: Exp5Schedule,
    history: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": phase,
        "dit": dit_state,
        "bridge": bridge_state,
        "bridge_ckpt": bridge_ckpt,
        "bridge_initialization": "exp1_functional_checkpoint",
        "config": cfg,
        "args": vars(args),
        "schedule": schedule.as_dict(),
        "history": history,
        "objective": {
            "mode": "exp5_staged_decoupled",
            "phase_a": "frozen bridge; DiT flow, teacher-output distillation, and preservation",
            "phase_b": "decoupled DiT and bridge optimizers; no flow gradient enters bridge",
            "phase_c": "frozen bridge; DiT consolidation and final cooldown",
            "bridge_representation_distillation": True,
            "frozen_teacher_bridge_functional_distillation": True,
            "teacher_forcing_previous_latents": True,
        },
        "parallel": {"backend": args.parallel, "world_size": world_size},
    }


def save_checkpoint(
    *,
    step: int,
    phase: str,
    dit_model: torch.nn.Module,
    bridge_model: torch.nn.Module,
    dit_optimizer: torch.optim.Optimizer,
    bridge_optimizer: torch.optim.Optimizer,
    sample_generator: torch.Generator,
    output_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    schedule: Exp5Schedule,
    history: list[dict[str, Any]],
    ctx,
    save_archive: bool,
    save_final: bool,
) -> None:
    dit_state = full_state_dict(dit_model, args.parallel)
    bridge_state = full_state_dict(bridge_model, args.parallel)
    dit_optimizer_state = full_optimizer_state_dict(dit_model, dit_optimizer, args.parallel)
    bridge_optimizer_state = full_optimizer_state_dict(bridge_model, bridge_optimizer, args.parallel)
    rng_states = gather_rng_states(capture_rng_state(sample_generator), ctx.world_size)

    if ctx.is_main:
        payload = model_payload(
            step=step,
            phase=phase,
            dit_state=dit_state,
            bridge_state=bridge_state,
            bridge_ckpt=args.bridge_ckpt,
            cfg=cfg,
            args=args,
            schedule=schedule,
            history=history,
            world_size=ctx.world_size,
        )
        latest_payload = {
            **payload,
            "dit_optimizer": dit_optimizer_state,
            "bridge_optimizer": bridge_optimizer_state,
            "rng_states": rng_states,
            "checkpoint_kind": "resumable_latest",
        }
        atomic_torch_save(latest_payload, output_dir / "neodragon_exp5_latest.pt")
        if save_archive:
            archive_payload = {**payload, "checkpoint_kind": "model_archive"}
            atomic_torch_save(
                archive_payload,
                output_dir / f"neodragon_exp5_step{step:06d}.pt",
            )
        if save_final:
            final_payload = {**payload, "checkpoint_kind": "final_model"}
            atomic_torch_save(final_payload, output_dir / "neodragon_exp5_final.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        rank0_print(
            ctx,
            f"Saved Exp5 checkpoint step={step} phase={phase} "
            f"latest=true archive={save_archive} final={save_final}",
        )
    barrier()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-phase NeoDragon Exp5 trainer.")
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", default="data/openvid_neodragon_2s_latents/latent_manifest.csv")
    parser.add_argument("--bridge-ckpt", required=True, help="Exp1 functional bridge checkpoint.")
    parser.add_argument("--output-dir", default="output/neo_exp5_staged")
    parser.add_argument("--resume", default="auto", help="auto, none, or an Exp5 latest checkpoint path.")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--steps", type=int, default=255_000)
    parser.add_argument("--phase-a-steps", type=int, default=10_000)
    parser.add_argument("--phase-b-steps", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp"], default="fsdp")
    parser.add_argument("--train-last-n-blocks", type=int, default=0)
    parser.add_argument("--dit-lr", type=float, default=3e-6)
    parser.add_argument("--bridge-lr", type=float, default=1e-6)
    parser.add_argument("--dit-warmup-steps", type=int, default=2_000)
    parser.add_argument("--bridge-warmup-steps", type=int, default=2_000)
    parser.add_argument("--bridge-cooldown-steps", type=int, default=10_000)
    parser.add_argument("--final-cooldown-steps", type=int, default=20_000)
    parser.add_argument("--flow-start-weight", type=float, default=0.05)
    parser.add_argument("--flow-weight", type=float, default=0.3)
    parser.add_argument("--flow-final-weight", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-cos-weight", type=float, default=0.1)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--preservation-cos-weight", type=float, default=0.05)
    parser.add_argument("--preservation-every", type=int, default=2)
    parser.add_argument("--bridge-repr-weight", type=float, default=0.5)
    parser.add_argument("--bridge-raw-token-weight", type=float, default=0.25)
    parser.add_argument("--bridge-normalized-token-weight", type=float, default=1.0)
    parser.add_argument("--bridge-cos-weight", type=float, default=0.5)
    parser.add_argument("--bridge-token-norm-weight", type=float, default=0.1)
    parser.add_argument("--bridge-pooled-weight", type=float, default=0.25)
    parser.add_argument("--bridge-pooled-cos-weight", type=float, default=0.2)
    parser.add_argument("--bridge-relational-weight", type=float, default=0.1)
    parser.add_argument("--bridge-functional-weight", type=float, default=1.0)
    parser.add_argument("--bridge-functional-cos-weight", type=float, default=0.1)
    parser.add_argument("--bridge-functional-every", type=int, default=2)
    parser.add_argument("--bridge-functional-batch-size", type=int, default=1)
    parser.add_argument("--caption-variant-columns", default="caption_short,caption_medium,caption_long")
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--diagnostic-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-latest-every", type=int, default=5_000)
    parser.add_argument("--save-archive-every", type=int, default=20_000)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-run-steps",
        type=int,
        default=-1,
        help="Stop cleanly after this many steps in the current invocation; intended for resume smoke tests.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "preservation_every": args.preservation_every,
        "bridge_functional_every": args.bridge_functional_every,
        "bridge_functional_batch_size": args.bridge_functional_batch_size,
        "log_every": args.log_every,
        "save_latest_every": args.save_latest_every,
        "save_archive_every": args.save_archive_every,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"These values must be positive: {invalid}")
    weights = {
        name: value
        for name, value in vars(args).items()
        if (name.endswith("_weight") or name.endswith("_lr")) and isinstance(value, (int, float))
    }
    negative = {name: value for name, value in weights.items() if value < 0.0}
    if negative:
        raise ValueError(f"Loss weights and learning rates must be non-negative: {negative}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    schedule = Exp5Schedule(
        total_steps=args.steps,
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        dit_warmup_steps=args.dit_warmup_steps,
        final_cooldown_steps=args.final_cooldown_steps,
        bridge_warmup_steps=args.bridge_warmup_steps,
        bridge_cooldown_steps=args.bridge_cooldown_steps,
        flow_start_weight=args.flow_start_weight,
        flow_peak_weight=args.flow_weight,
        flow_final_weight=args.flow_final_weight,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")
    ctx = setup_distributed()
    if args.parallel in {"ddp", "fsdp"} and not ctx.is_distributed:
        rank0_print(ctx, f"Warning: --parallel={args.parallel} with world_size=1; using --parallel=none.")
        args.parallel = "none"

    output_dir = Path(args.output_dir).expanduser()
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    resume_path = resolve_resume_path(args.resume, output_dir)
    resume_payload = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None
        else None
    )
    start_step = int(resume_payload.get("step", 0)) if resume_payload is not None else 0
    if resume_payload is not None:
        validate_resume_compatibility(resume_payload, args, schedule, ctx.world_size)
    if start_step >= args.steps:
        rank0_print(ctx, f"Exp5 already complete at step={start_step}; requested steps={args.steps}.")
        cleanup_distributed()
        return

    random.seed(args.seed + ctx.rank)
    np.random.seed(args.seed + ctx.rank)
    torch.manual_seed(args.seed + ctx.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + ctx.rank)

    cfg = load_config(args.config)
    if args.dtype:
        cfg.backend.dtype = args.dtype
        cfg.train.dtype = args.dtype
    dtype = dtype_from_name(cfg.backend.dtype)
    if ctx.device.type == "cpu":
        dtype = torch.float32

    caption_columns = parse_csv_list(args.caption_variant_columns)
    caption_weights = parse_float_list(args.caption_variant_weights)
    if len(caption_columns) != len(caption_weights):
        raise ValueError("caption variant columns and weights must have equal lengths.")
    dataset = VideoPromptDataset(
        args.manifest,
        max_samples=args.max_samples,
        caption_aug=True,
        caption_variant_columns=caption_columns,
        caption_variant_weights=caption_weights,
        caption_fallback_column=args.caption_fallback_column,
    )
    if not dataset.has_latents:
        raise ValueError("Exp5 requires an offline latent manifest containing latent_path.")
    sampler = OffsetDistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda batch: collate_video_batch(
            batch,
            num_frames=cfg.data.frame_num,
            height=cfg.data.height,
            width=cfg.data.width,
            target_fps=args.target_fps,
            latent_root=Path(args.manifest).expanduser().parent,
            use_latents=True,
        ),
    )
    full_steps_per_epoch = math.ceil(sampler.num_samples / args.batch_size)
    batches = iter_batches_from_step(
        loader,
        sampler,
        completed_steps=start_step,
        full_steps_per_epoch=full_steps_per_epoch,
        batch_size=args.batch_size,
    )

    rank0_print(
        ctx,
        f"Exp5 data: samples={len(dataset)} world_size={ctx.world_size} batch_per_gpu={args.batch_size} "
        f"global_batch={ctx.world_size * args.batch_size} steps_per_epoch={full_steps_per_epoch} "
        f"planned_epochs={args.steps / full_steps_per_epoch:.4f}",
    )
    rank0_print(ctx, f"Exp5 schedule: {json.dumps(schedule.as_dict(), sort_keys=True)}")

    dit, _, scheduler, prompt_modifier = load_neodragon_train_modules(
        cfg,
        ctx.device,
        dtype,
        load_vae=False,
    )
    if resume_payload is not None:
        bridge = load_bridge(cfg, None, ctx.device, dtype, trainable=True)
        dit.load_state_dict(resume_payload["dit"], strict=True)
        bridge.load_state_dict(resume_payload["bridge"], strict=True)
    else:
        bridge = load_bridge(cfg, args.bridge_ckpt, ctx.device, dtype, trainable=True)
    freeze_dit_for_last_n_blocks(dit, args.train_last_n_blocks)
    dit.train()
    bridge.train()

    dit_model = wrap_dit(
        dit,
        parallel=args.parallel,
        device=ctx.device,
        local_rank=ctx.local_rank,
    )
    bridge_model = wrap_bridge(
        bridge,
        parallel=args.parallel,
        device=ctx.device,
        local_rank=ctx.local_rank,
    )
    dit_optimizer = torch.optim.AdamW(
        [parameter for parameter in dit_model.parameters() if parameter.requires_grad],
        lr=args.dit_lr,
        weight_decay=0.0,
    )
    bridge_optimizer = torch.optim.AdamW(
        [parameter for parameter in bridge_model.parameters() if parameter.requires_grad],
        lr=args.bridge_lr,
        weight_decay=0.0,
    )

    teacher_text, teacher_context_adapter, teacher_dit = load_neodragon_teacher_modules(
        cfg,
        ctx.device,
        dtype,
    )

    sample_generator = torch.Generator(device=ctx.device)
    sample_generator.manual_seed(args.seed + 100_000 + ctx.rank)
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        history = list(resume_payload.get("history") or [])
        if not args.reset_optimizer:
            if "dit_optimizer" not in resume_payload or "bridge_optimizer" not in resume_payload:
                raise ValueError("Resume checkpoint lacks optimizer state; pass --reset-optimizer to continue anyway.")
            load_full_optimizer_state_dict(
                dit_model,
                dit_optimizer,
                resume_payload.get("dit_optimizer"),
                args.parallel,
                is_main=ctx.is_main,
            )
            load_full_optimizer_state_dict(
                bridge_model,
                bridge_optimizer,
                resume_payload.get("bridge_optimizer"),
                args.parallel,
                is_main=ctx.is_main,
            )
        rng_states = resume_payload.get("rng_states") or []
        if len(rng_states) == ctx.world_size:
            restore_rng_state(rng_states[ctx.rank], sample_generator)
        else:
            rank0_print(
                ctx,
                f"Warning: RNG state world_size={len(rng_states)} differs from current world_size={ctx.world_size}; "
                "using deterministic rank seeds.",
            )
        rank0_print(
            ctx,
            f"Resumed Exp5 from {resume_path}: step={start_step} phase={resume_payload.get('phase')} "
            f"reset_optimizer={args.reset_optimizer}",
        )
        del resume_payload
        gc.collect()

    from neodragon.utils.generation_utils import _get_pyramid_latent, _prepare_past_condition_latents

    window = MetricWindow()
    run_end_step = args.steps
    if args.max_run_steps > 0:
        run_end_step = min(args.steps, start_step + args.max_run_steps)
    pbar = tqdm(
        range(start_step + 1, run_end_step + 1),
        desc="Train NeoDragon Exp5",
        disable=not ctx.is_main,
        initial=start_step,
        total=args.steps,
    )
    previous_phase = None
    for step in pbar:
        phase = schedule.phase(step)
        bridge_active = phase.train_bridge
        if phase.name != previous_phase:
            rank0_print(ctx, f"Exp5 phase transition: step={step} phase={phase.name}")
            previous_phase = phase.name

        dit_lr = args.dit_lr * schedule.dit_lr_scale(step)
        bridge_lr = args.bridge_lr * schedule.bridge_lr_scale(step)
        flow_weight = schedule.flow_weight(step)
        set_optimizer_lr(dit_optimizer, dit_lr)
        set_optimizer_lr(bridge_optimizer, bridge_lr)
        dit_model.train()
        bridge_model.train(bridge_active)

        batch = next(batches)
        prompts = [str(prompt) + prompt_modifier for prompt in batch["prompt"]]
        if bridge_active:
            bridge_tokens, bridge_mask, bridge_pooled = bridge_model(prompts)
        else:
            with torch.no_grad():
                bridge_tokens, bridge_mask, bridge_pooled = bridge_model(prompts)

        with torch.no_grad():
            latents = batch["latents"].to(device=ctx.device, dtype=dtype, non_blocking=True)
            teacher_tokens, teacher_mask, teacher_pooled = teacher_text(prompts, ctx.device)
            teacher_tokens = teacher_context_adapter(teacher_tokens)

        latent_t = int(latents.shape[2])
        if latent_t < 2:
            raise RuntimeError(f"Need at least two latent frames, got latent_t={latent_t}")
        unit_index = int(torch.randint(1, latent_t, (1,), device=ctx.device, generator=sample_generator).item())
        stage = int(
            torch.randint(
                0,
                scheduler.config.stages,
                (1,),
                device=ctx.device,
                generator=sample_generator,
            ).item()
        )
        past_units = [latents[:, :, index : index + 1] for index in range(unit_index)]
        past_conditions = _prepare_past_condition_latents(
            past_units,
            num_stages=scheduler.config.stages,
            do_classifier_free_guidance=False,
        )
        clean_full = latents[:, :, unit_index : unit_index + 1]
        clean_stage = _get_pyramid_latent(clean_full, scheduler.config.stages)[stage].to(dtype=dtype)
        noise = torch.randn(
            clean_stage.shape,
            device=ctx.device,
            dtype=dtype,
            generator=sample_generator,
        )
        timestep_index = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (clean_stage.shape[0],),
            device=ctx.device,
            generator=sample_generator,
        )
        sigmas = scheduler.sigmas_per_stage[stage].to(device=ctx.device, dtype=dtype)[timestep_index]
        timestep = scheduler.timesteps_per_stage[stage].to(device=ctx.device, dtype=dtype)[timestep_index]
        while sigmas.dim() < clean_stage.dim():
            sigmas = sigmas.view(-1, *([1] * (clean_stage.dim() - 1)))
        noisy = sigmas * noise + (1.0 - sigmas) * clean_stage
        target_flow = noise - clean_stage
        stage_input = past_conditions[stage] + [noisy]

        # Flow and joint output distillation update DiT only. This prevents the
        # DiT from co-adapting with, or overwriting, the aligned Exp1 bridge.
        prediction = dit_model(
            sample=[stage_input],
            encoder_hidden_states=bridge_tokens.detach(),
            encoder_attention_mask=bridge_mask.detach(),
            pooled_projections=bridge_pooled.detach(),
            timestep_ratio=timestep,
        )[0]
        flow_loss = F.mse_loss(prediction.float(), target_flow.float())
        with torch.no_grad():
            teacher_prediction = teacher_dit(
                sample=[stage_input],
                encoder_hidden_states=teacher_tokens,
                encoder_attention_mask=teacher_mask,
                pooled_projections=teacher_pooled,
                timestep_ratio=timestep,
            )[0]
        distill_loss = F.mse_loss(prediction.float(), teacher_prediction.float())
        distill_cos_loss = flat_cosine_distance(prediction, teacher_prediction)

        preservation_active = (step - 1) % args.preservation_every == 0
        preservation_loss = prediction.new_zeros(())
        preservation_cos_loss = prediction.new_zeros(())
        if preservation_active:
            teacher_condition_prediction = dit_model(
                sample=[stage_input],
                encoder_hidden_states=teacher_tokens,
                encoder_attention_mask=teacher_mask,
                pooled_projections=teacher_pooled,
                timestep_ratio=timestep,
            )[0]
            preservation_loss = F.mse_loss(
                teacher_condition_prediction.float(),
                teacher_prediction.float(),
            )
            preservation_cos_loss = flat_cosine_distance(
                teacher_condition_prediction,
                teacher_prediction,
            )
        preservation_frequency_scale = float(args.preservation_every) if preservation_active else 0.0
        dit_loss = (
            flow_weight * flow_loss
            + args.distill_weight * distill_loss
            + args.distill_cos_weight * distill_cos_loss
            + preservation_frequency_scale
            * (
                args.preservation_weight * preservation_loss
                + args.preservation_cos_weight * preservation_cos_loss
            )
        )

        bridge_repr_loss = prediction.new_zeros(())
        bridge_repr_losses = {
            name: prediction.new_zeros(())
            for name in [
                "raw_token",
                "normalized_token",
                "token_cosine",
                "token_norm",
                "pooled_mse",
                "pooled_cosine",
                "relational",
            ]
        }
        bridge_functional_active = False
        bridge_functional_loss = prediction.new_zeros(())
        bridge_functional_cos_loss = prediction.new_zeros(())
        bridge_loss = prediction.new_zeros(())
        if bridge_active:
            bridge_repr_losses = bridge_representation_losses(
                bridge_tokens,
                teacher_tokens,
                teacher_mask,
                bridge_pooled,
                teacher_pooled,
            )
            if args.bridge_relational_weight > 0.0:
                global_bridge_tokens = _gather_with_gradient(bridge_tokens)
                global_teacher_tokens = _gather_detached(teacher_tokens)
                global_teacher_mask = _gather_detached(teacher_mask)
                bridge_repr_losses["relational"] = relational_cosine(
                    global_bridge_tokens,
                    global_teacher_tokens,
                    global_teacher_mask,
                )
            bridge_repr_loss = weighted_loss_sum(
                bridge_repr_losses,
                {
                    "raw_token": args.bridge_raw_token_weight,
                    "normalized_token": args.bridge_normalized_token_weight,
                    "token_cosine": args.bridge_cos_weight,
                    "token_norm": args.bridge_token_norm_weight,
                    "pooled_mse": args.bridge_pooled_weight,
                    "pooled_cosine": args.bridge_pooled_cos_weight,
                    "relational": args.bridge_relational_weight,
                },
            )
            bridge_functional_active = (step - 1) % args.bridge_functional_every == 0
            if bridge_functional_active:
                effect_batch = min(args.bridge_functional_batch_size, bridge_tokens.shape[0])
                effect_stage_input = [value[:effect_batch] for value in stage_input]
                bridge_teacher_prediction = teacher_dit(
                    sample=[effect_stage_input],
                    encoder_hidden_states=bridge_tokens[:effect_batch],
                    encoder_attention_mask=bridge_mask[:effect_batch],
                    pooled_projections=bridge_pooled[:effect_batch],
                    timestep_ratio=timestep[:effect_batch],
                )[0]
                bridge_functional_loss = F.mse_loss(
                    bridge_teacher_prediction.float(),
                    teacher_prediction[:effect_batch].float(),
                )
                bridge_functional_cos_loss = flat_cosine_distance(
                    bridge_teacher_prediction,
                    teacher_prediction[:effect_batch],
                )
            functional_frequency_scale = (
                float(args.bridge_functional_every) if bridge_functional_active else 0.0
            )
            bridge_loss = (
                args.bridge_repr_weight * bridge_repr_loss
                + functional_frequency_scale
                * (
                    args.bridge_functional_weight * bridge_functional_loss
                    + args.bridge_functional_cos_weight * bridge_functional_cos_loss
                )
            )

        dit_optimizer.zero_grad(set_to_none=True)
        bridge_optimizer.zero_grad(set_to_none=True)
        dit_loss.backward()
        dit_grad_norm = clip_grad_norm(dit_model, args.clip_grad_norm, args.parallel)
        dit_optimizer.step()
        bridge_grad_norm = prediction.new_zeros(())
        if bridge_active:
            bridge_loss.backward()
            bridge_grad_norm = clip_grad_norm(bridge_model, args.clip_grad_norm, args.parallel)
            bridge_optimizer.step()

        diagnostic_active = args.diagnostic_every > 0 and step % args.diagnostic_every == 0
        diagnostic_correct_flow = prediction.new_zeros(())
        diagnostic_shuffled_flow = prediction.new_zeros(())
        diagnostic_text_sensitivity = prediction.new_zeros(())
        diagnostic_offdiag_cos = prediction.new_zeros(())
        if diagnostic_active:
            shuffled_tokens = _shift_condition_across_global_batch(bridge_tokens)
            shuffled_mask = _shift_condition_across_global_batch(bridge_mask)
            shuffled_pooled = _shift_condition_across_global_batch(bridge_pooled)
            if shuffled_tokens is not None and shuffled_mask is not None and shuffled_pooled is not None:
                with torch.no_grad():
                    correct_prediction = dit_model(
                        sample=[stage_input],
                        encoder_hidden_states=bridge_tokens.detach(),
                        encoder_attention_mask=bridge_mask.detach(),
                        pooled_projections=bridge_pooled.detach(),
                        timestep_ratio=timestep,
                    )[0]
                    shuffled_prediction = dit_model(
                        sample=[stage_input],
                        encoder_hidden_states=shuffled_tokens,
                        encoder_attention_mask=shuffled_mask,
                        pooled_projections=shuffled_pooled,
                        timestep_ratio=timestep,
                    )[0]
                    diagnostic_correct_flow = F.mse_loss(correct_prediction.float(), target_flow.float())
                    diagnostic_shuffled_flow = F.mse_loss(shuffled_prediction.float(), target_flow.float())
                    diagnostic_text_sensitivity = F.mse_loss(
                        correct_prediction.float(),
                        shuffled_prediction.float(),
                    )
                    diagnostic_offdiag_cos = _condition_offdiag_cosine(
                        _gather_detached(bridge_tokens),
                        _gather_detached(bridge_mask),
                    )

        total_loss = dit_loss.detach() + bridge_loss.detach()
        window.add("loss", total_loss)
        window.add("dit_loss", dit_loss)
        window.add("bridge_loss", bridge_loss, active=bridge_active)
        window.add("flow_loss", flow_loss)
        window.add("distill_loss", distill_loss)
        window.add("distill_cos_loss", distill_cos_loss)
        window.add("preservation_loss", preservation_loss, active=preservation_active)
        window.add("preservation_cos_loss", preservation_cos_loss, active=preservation_active)
        window.add("bridge_repr_loss", bridge_repr_loss, active=bridge_active)
        for name, value in bridge_repr_losses.items():
            window.add(f"bridge_{name}", value, active=bridge_active)
        window.add(
            "bridge_functional_loss",
            bridge_functional_loss,
            active=bridge_functional_active,
        )
        window.add(
            "bridge_functional_cos_loss",
            bridge_functional_cos_loss,
            active=bridge_functional_active,
        )
        window.add("dit_grad_norm", dit_grad_norm)
        window.add("bridge_grad_norm", bridge_grad_norm, active=bridge_active)
        window.add("diagnostic_correct_flow", diagnostic_correct_flow, active=diagnostic_active)
        window.add("diagnostic_shuffled_flow", diagnostic_shuffled_flow, active=diagnostic_active)
        window.add("diagnostic_text_sensitivity", diagnostic_text_sensitivity, active=diagnostic_active)
        window.add("diagnostic_offdiag_cos", diagnostic_offdiag_cos, active=diagnostic_active)

        if step % args.log_every == 0 or step == start_step + 1:
            item: dict[str, Any] = {
                "step": step,
                "phase": phase.name,
                "epoch": step / full_steps_per_epoch,
                "dit_lr": dit_lr,
                "bridge_lr": bridge_lr,
                "flow_weight": flow_weight,
                "bridge_active": bridge_active,
                "unit_index": unit_index,
                "stage": stage,
                "latent_t": latent_t,
                "world_size": ctx.world_size,
                "global_batch": ctx.world_size * args.batch_size,
                "gpu_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(ctx.device) / (1024**3))
                    if ctx.device.type == "cuda"
                    else 0.0
                ),
                **window.distributed_means(ctx.device),
            }
            if ctx.is_main:
                history.append(item)
                pbar.set_postfix(
                    phase=phase.name.split("_")[0],
                    loss=f"{item.get('loss', 0.0):.4f}",
                    flow=f"{item.get('flow_loss', 0.0):.4f}",
                    dist=f"{item.get('distill_loss', 0.0):.4f}",
                    bfunc=f"{item.get('bridge_functional_loss', 0.0):.4f}",
                )

        save_latest = (
            step % args.save_latest_every == 0
            or step == args.steps
            or step == run_end_step
        )
        if save_latest:
            save_checkpoint(
                step=step,
                phase=phase.name,
                dit_model=dit_model,
                bridge_model=bridge_model,
                dit_optimizer=dit_optimizer,
                bridge_optimizer=bridge_optimizer,
                sample_generator=sample_generator,
                output_dir=output_dir,
                cfg=cfg,
                args=args,
                schedule=schedule,
                history=history,
                ctx=ctx,
                save_archive=step % args.save_archive_every == 0,
                save_final=step == args.steps,
            )

    if run_end_step == args.steps:
        rank0_print(ctx, f"Exp5 complete: {output_dir / 'neodragon_exp5_final.pt'}")
    else:
        rank0_print(
            ctx,
            f"Exp5 paused cleanly at step={run_end_step}; resume with the same command and --resume auto.",
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
