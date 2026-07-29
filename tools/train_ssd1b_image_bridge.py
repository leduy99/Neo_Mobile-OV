#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVSSD1BImageBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.neodragon_objectives import linear_ramp
from new_mobile_ov.training.ssd1b_distillation import (
    SSD1BClosedLoopResult,
    SSD1BFunctionalResult,
    SSD1BFrozenTeacher,
    SSD1BFrozenUNetController,
    SSD1BRolloutResult,
    ssd1b_representation_losses,
)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


class CaptionDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        variant_columns: list[str],
        variant_weights: list[float],
        fallback_column: str,
        max_samples: int,
    ) -> None:
        path = Path(path)
        if path.suffix.lower() not in {".csv", ".tsv"}:
            prompts = [clean_text(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.items = [[(prompt, 1.0)] for prompt in prompts if prompt]
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, low_memory=False)
            fallback = fallback_column if fallback_column in frame.columns else None
            if fallback is None:
                fallback = next((name for name in ("caption", "prompt", "text") if name in frame.columns), None)
            if fallback is None:
                raise ValueError(f"{path} has no caption, prompt, or text column.")
            items: list[list[tuple[str, float]]] = []
            for _, row in frame.iterrows():
                variants = []
                for name, weight in zip(variant_columns, variant_weights):
                    if name in frame.columns:
                        value = clean_text(row.get(name))
                        if value:
                            variants.append((value, float(weight)))
                if not variants:
                    value = clean_text(row.get(fallback))
                    if value:
                        variants.append((value, 1.0))
                if variants:
                    items.append(variants)
            self.items = items
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"No valid prompts found in {path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> str:
        variants = self.items[index]
        if len(variants) == 1:
            return variants[0][0]
        return random.choices(
            [value for value, _ in variants],
            weights=[weight for _, weight in variants],
            k=1,
        )[0]


def dtype_from_name(value: str) -> torch.dtype:
    value = str(value).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value!r}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def unwrap(model: torch.nn.Module) -> MobileOVSSD1BImageBridge:
    return getattr(model, "module", model)


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
    return float(final_scale) + (1.0 - float(final_scale)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def representation_loss(
    losses: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.objective_version == "v2":
        clip_l = (
            losses["clip_l_content_normalized_mse"]
            + losses["clip_l_content_cosine"]
            + args.eos_weight
            * (
                losses["clip_l_eos_normalized_mse"]
                + losses["clip_l_eos_cosine"]
            )
            + args.padding_weight
            * (
                losses["clip_l_padding_normalized_mse"]
                + losses["clip_l_padding_cosine"]
            )
        )
        clip_big_g = (
            losses["clip_big_g_content_normalized_mse"]
            + losses["clip_big_g_content_cosine"]
            + args.eos_weight
            * (
                losses["clip_big_g_eos_normalized_mse"]
                + losses["clip_big_g_eos_cosine"]
            )
            + args.padding_weight
            * (
                losses["clip_big_g_padding_normalized_mse"]
                + losses["clip_big_g_padding_cosine"]
            )
        )
        pooled = losses["pooled_mse"] + losses["pooled_cosine"]
        norm = 0.5 * (
            losses["clip_l_content_norm"]
            + losses["clip_big_g_content_norm"]
        )
        geometry = (
            losses["clip_l_geometry"]
            + losses["clip_big_g_geometry"]
            + losses["pooled_geometry"]
        ) / 3.0
        retrieval = (
            losses["clip_l_retrieval"]
            + losses["clip_big_g_retrieval"]
            + losses["pooled_retrieval"]
        ) / 3.0
        variance = (
            losses["clip_l_variance"]
            + losses["clip_big_g_variance"]
            + losses["pooled_variance"]
        ) / 3.0
        return (
            args.clip_l_weight * clip_l
            + args.clip_big_g_weight * clip_big_g
            + args.pooled_weight * pooled
            + args.norm_weight * norm
            + args.geometry_weight * geometry
            + args.retrieval_weight * retrieval
            + args.variance_weight * variance
        )

    clip_l = losses["clip_l_normalized_mse"] + losses["clip_l_cosine"]
    clip_big_g = losses["clip_big_g_normalized_mse"] + losses["clip_big_g_cosine"]
    pooled = losses["pooled_mse"] + losses["pooled_cosine"]
    norm = 0.5 * (losses["clip_l_norm"] + losses["clip_big_g_norm"])
    geometry = (
        losses["clip_l_geometry"]
        + losses["clip_big_g_geometry"]
        + losses["pooled_geometry"]
    ) / 3.0
    return (
        args.clip_l_weight * clip_l
        + args.clip_big_g_weight * clip_big_g
        + args.pooled_weight * pooled
        + args.norm_weight * norm
        + args.geometry_weight * geometry
    )


def representation_scale(step: int, args: argparse.Namespace) -> float:
    if args.objective_version != "v2" or step < args.closed_loop_start_step:
        return 1.0
    if args.representation_decay_steps <= 0:
        return float(args.representation_final_scale)
    progress = min(
        max(
            (step - args.closed_loop_start_step + 1)
            / float(args.representation_decay_steps),
            0.0,
        ),
        1.0,
    )
    return 1.0 + (float(args.representation_final_scale) - 1.0) * progress


def load_hard_prompts(path: str | None) -> list[str]:
    if not path:
        return []
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Hard-prompt file does not exist: {prompt_path}")
    return [
        clean_text(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if clean_text(line)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill SSD1B's two CLIP encoders into the Mobile-OV Image Bridge."
    )
    parser.add_argument("--config", default="configs/mobile_ov_ssd1b_image_bridge.yaml")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", default="output/ssd1b_image_bridge_distill")
    parser.add_argument("--target-step", type=int, default=100000)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--objective-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--caption-variant-columns", default="caption_short,caption_medium,caption_long")
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument(
        "--append-prompt-modifier",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prompt-modifier-prob",
        type=float,
        default=None,
        help="Per-prompt modifier probability. Overrides --append-prompt-modifier.",
    )
    parser.add_argument("--hard-prompts", default=None)
    parser.add_argument("--hard-prompt-prob", type=float, default=0.0)

    parser.add_argument("--clip-l-weight", type=float, default=0.25)
    parser.add_argument("--clip-big-g-weight", type=float, default=0.25)
    parser.add_argument("--pooled-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=0.5)
    parser.add_argument("--norm-weight", type=float, default=0.1)
    parser.add_argument("--eos-weight", type=float, default=0.5)
    parser.add_argument("--padding-weight", type=float, default=0.15)
    parser.add_argument("--retrieval-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--representation-final-scale", type=float, default=0.35)
    parser.add_argument("--representation-decay-steps", type=int, default=15000)
    parser.add_argument("--functional-weight", type=float, default=1.0)
    parser.add_argument("--functional-cos-weight", type=float, default=0.1)
    parser.add_argument("--functional-start-step", type=int, default=5001)
    parser.add_argument("--functional-ramp-steps", type=int, default=5000)
    parser.add_argument("--functional-batch-size", type=int, default=1)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--rollout-cos-weight", type=float, default=0.1)
    parser.add_argument("--rollout-transition-weight", type=float, default=0.5)
    parser.add_argument("--rollout-start-step", type=int, default=10001)
    parser.add_argument("--rollout-ramp-steps", type=int, default=5000)
    parser.add_argument("--rollout-every", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--closed-loop-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-prediction-weight", type=float, default=0.5)
    parser.add_argument("--closed-loop-cos-weight", type=float, default=0.1)
    parser.add_argument("--closed-loop-transition-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-terminal-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-start-step", type=int, default=25001)
    parser.add_argument("--closed-loop-ramp-steps", type=int, default=5000)
    parser.add_argument("--closed-loop-every", type=int, default=4)
    parser.add_argument("--closed-loop-batch-size", type=int, default=1)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-latest-every", type=int, default=5000)
    parser.add_argument("--save-archive-every", type=int, default=10000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "target_step": args.target_step,
        "batch_size": args.batch_size,
        "functional_batch_size": args.functional_batch_size,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_every": args.rollout_every,
        "closed_loop_every": args.closed_loop_every,
        "closed_loop_batch_size": args.closed_loop_batch_size,
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
        if name.endswith("_weight")
    }
    invalid_weights = {name: value for name, value in weights.items() if value < 0}
    if invalid_weights:
        raise ValueError(f"Loss weights must be non-negative: {invalid_weights}")
    probabilities = {
        "hard_prompt_prob": args.hard_prompt_prob,
    }
    if args.prompt_modifier_prob is not None:
        probabilities["prompt_modifier_prob"] = args.prompt_modifier_prob
    invalid_probabilities = {
        name: value
        for name, value in probabilities.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid_probabilities:
        raise ValueError(f"Probabilities must be within [0, 1]: {invalid_probabilities}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    context = setup_distributed()
    if context.world_size > 1 and context.device.type != "cuda":
        raise RuntimeError("Distributed SSD1B distillation requires CUDA.")

    seed = args.seed + context.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=context.device).manual_seed(seed + 100000)

    config = load_config(args.config)
    inference_dtype = dtype_from_name(args.dtype)
    if context.device.type == "cpu":
        inference_dtype = torch.float32
    output_dir = Path(args.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    columns = split_csv(args.caption_variant_columns)
    weights = [float(value) for value in split_csv(args.caption_variant_weights)]
    if len(columns) != len(weights):
        raise ValueError("Caption variant columns and weights must have the same length.")
    dataset = CaptionDataset(
        args.prompts,
        variant_columns=columns,
        variant_weights=weights,
        fallback_column=args.caption_fallback_column,
        max_samples=args.max_samples,
    )
    hard_prompts = load_hard_prompts(args.hard_prompts)
    if args.hard_prompt_prob > 0 and not hard_prompts:
        raise ValueError("--hard-prompt-prob requires a non-empty --hard-prompts file.")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if context.is_distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is smaller than one global training batch.")

    rank0_print(
        context,
        "SSD1B Image Bridge distillation: "
        f"world_size={context.world_size} batch_per_gpu={args.batch_size} "
        f"global_batch={context.world_size * args.batch_size} rows={len(dataset)} "
        f"dtype={inference_dtype} target_step={args.target_step}",
    )

    # Stagger multi-process file reads without changing distributed semantics.
    if context.device.type == "cuda":
        time.sleep(2 * context.local_rank)
    teacher = SSD1BFrozenTeacher(config, context.device, inference_dtype)
    controller = SSD1BFrozenUNetController(config, context.device, inference_dtype)
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    bridge = MobileOVSSD1BImageBridge(
        config.bridge,
        config.image_bridge,
        device=context.device,
        dtype=inference_dtype,
    ).train()
    bridge.promote_trainable_parameters_to_fp32()
    trainable_count = sum(
        parameter.numel()
        for parameter in bridge.parameters()
        if parameter.requires_grad
    )
    rank0_print(
        context,
        f"Image Bridge ready: trainable_params={trainable_count:,} "
        "master_dtype=float32 outputs=77x768,77x1280,1280",
    )

    latest_path = output_dir / "ssd1b_image_bridge_latest.pt"
    resume_path = None
    if args.resume == "auto" and latest_path.is_file():
        resume_path = latest_path
    elif args.resume not in {"", "none", "auto"}:
        resume_path = Path(args.resume)
    checkpoint = None
    current_step = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        bridge.load_trainable_state_dict(checkpoint["image_bridge"])
        current_step = int(checkpoint.get("step", 0))
        rank0_print(context, f"Resumed Image Bridge from {resume_path} at step={current_step}")
    if current_step >= args.target_step:
        rank0_print(context, f"Nothing to train: step={current_step} target={args.target_step}")
        cleanup_distributed()
        return

    bridge_model: torch.nn.Module = bridge
    if context.is_distributed:
        bridge_model = DDP(
            bridge,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=False,
        )
    trainable = [parameter for parameter in bridge_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    history = list(checkpoint.get("history", [])) if checkpoint is not None else []

    epoch, skip_batch = divmod(current_step, len(loader))
    progress = tqdm(
        total=args.target_step,
        initial=current_step,
        desc="Train SSD1B Image Bridge",
        disable=not context.is_main,
    )
    while current_step < args.target_step:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_index, prompt_batch in enumerate(loader):
            if epoch * len(loader) + batch_index < current_step:
                continue
            if batch_index < skip_batch:
                continue
            current_step += 1
            prompts = [str(value) for value in prompt_batch]
            if hard_prompts:
                prompts = [
                    random.choice(hard_prompts)
                    if random.random() < args.hard_prompt_prob
                    else prompt
                    for prompt in prompts
                ]
            modifier_probability = args.prompt_modifier_prob
            if modifier_probability is None:
                modifier_probability = 1.0 if args.append_prompt_modifier else 0.0
            prompts = [
                prompt + DEFAULT_PROMPT_MODIFIER
                if random.random() < modifier_probability
                else prompt
                for prompt in prompts
            ]

            with torch.no_grad():
                teacher_condition = teacher.encode(prompts)

            autocast_enabled = context.device.type == "cuda" and inference_dtype != torch.float32
            with torch.autocast(
                device_type=context.device.type,
                dtype=inference_dtype,
                enabled=autocast_enabled,
            ):
                student_condition = bridge_model(prompts)
                if current_step == 1:
                    rank0_print(
                        context,
                        "Condition shapes: "
                        f"student_l={tuple(student_condition.clip_l_tokens.shape)} "
                        f"student_g={tuple(student_condition.clip_big_g_tokens.shape)} "
                        f"student_pool={tuple(student_condition.pooled.shape)} "
                        f"teacher_l={tuple(teacher_condition.clip_l_tokens.shape)} "
                        f"teacher_g={tuple(teacher_condition.clip_big_g_tokens.shape)} "
                        f"teacher_pool={tuple(teacher_condition.pooled.shape)}",
                    )
                component_losses = ssd1b_representation_losses(
                    student_condition,
                    teacher_condition,
                )
                repr_loss = representation_loss(component_losses, args)
                repr_scale = representation_scale(current_step, args)

                functional = SSD1BFunctionalResult(
                    mse=repr_loss.new_zeros(()),
                    cosine=repr_loss.new_zeros(()),
                    timestep=-1,
                )
                rollout = SSD1BRolloutResult(
                    prediction_mse=repr_loss.new_zeros(()),
                    prediction_cosine=repr_loss.new_zeros(()),
                    transition_mse=repr_loss.new_zeros(()),
                    per_step_mse=(),
                    calls=0,
                )
                closed_loop = SSD1BClosedLoopResult(
                    prediction_relative_mse=repr_loss.new_zeros(()),
                    prediction_cosine=repr_loss.new_zeros(()),
                    transition_relative_mse=repr_loss.new_zeros(()),
                    transition_cosine=repr_loss.new_zeros(()),
                    terminal_relative_mse=repr_loss.new_zeros(()),
                    per_step_transition_relative_mse=(),
                    calls=0,
                )
                mode = "representation"
                functional_scale = 0.0
                rollout_scale = 0.0
                closed_loop_scale = 0.0
                run_closed_loop = (
                    args.objective_version == "v2"
                    and current_step >= args.closed_loop_start_step
                    and (current_step - args.closed_loop_start_step)
                    % args.closed_loop_every
                    == 0
                )
                run_rollout = (
                    args.objective_version == "v1"
                    and
                    current_step >= args.rollout_start_step
                    and (current_step - args.rollout_start_step) % args.rollout_every == 0
                )
                if run_closed_loop:
                    mode = "closed_loop"
                    closed_loop_scale = linear_ramp(
                        current_step,
                        start_step=args.closed_loop_start_step,
                        ramp_steps=args.closed_loop_ramp_steps,
                    )
                    closed_loop = controller.closed_loop_loss(
                        student_condition,
                        teacher_condition,
                        batch_size=args.closed_loop_batch_size,
                        generator=generator,
                    )
                elif run_rollout:
                    mode = "rollout"
                    rollout_scale = linear_ramp(
                        current_step,
                        start_step=args.rollout_start_step,
                        ramp_steps=args.rollout_ramp_steps,
                    )
                    rollout = controller.rollout_loss(
                        student_condition,
                        teacher_condition,
                        batch_size=args.rollout_batch_size,
                        generator=generator,
                    )
                elif current_step >= args.functional_start_step:
                    mode = "functional"
                    functional_scale = linear_ramp(
                        current_step,
                        start_step=args.functional_start_step,
                        ramp_steps=args.functional_ramp_steps,
                    )
                    functional = controller.functional_loss(
                        student_condition,
                        teacher_condition,
                        batch_size=args.functional_batch_size,
                        generator=generator,
                        timestep_index=(
                            (current_step + context.rank) % 4
                            if args.objective_version == "v2"
                            else None
                        ),
                    )

                loss = repr_scale * repr_loss
                loss = loss + functional_scale * (
                    args.functional_weight * functional.mse
                    + args.functional_cos_weight * functional.cosine
                )
                loss = loss + rollout_scale * (
                    args.rollout_weight * rollout.prediction_mse
                    + args.rollout_cos_weight * rollout.prediction_cosine
                    + args.rollout_transition_weight * rollout.transition_mse
                )
                loss = loss + closed_loop_scale * args.closed_loop_weight * (
                    args.closed_loop_prediction_weight
                    * closed_loop.prediction_relative_mse
                    + args.closed_loop_cos_weight
                    * (
                        closed_loop.prediction_cosine
                        + closed_loop.transition_cosine
                    )
                    + args.closed_loop_transition_weight
                    * closed_loop.transition_relative_mse
                    + args.closed_loop_terminal_weight
                    * closed_loop.terminal_relative_mse
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
            optimizer.step()
            lr_scale = learning_rate_scale(
                current_step,
                total_steps=args.target_step,
                warmup_steps=args.lr_warmup_steps,
                final_scale=args.lr_final_scale,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale

            if (
                current_step % args.log_every == 0
                or current_step == 1
                or run_rollout
                or run_closed_loop
            ):
                item = {
                    "step": current_step,
                    "loss": scalar_mean(loss.detach(), context),
                    "representation": scalar_mean(repr_loss.detach(), context),
                    "representation_scale": repr_scale,
                    "clip_l_normalized_mse": scalar_mean(
                        component_losses["clip_l_normalized_mse"].detach(),
                        context,
                    ),
                    "clip_l_cosine": scalar_mean(component_losses["clip_l_cosine"].detach(), context),
                    "clip_big_g_normalized_mse": scalar_mean(
                        component_losses["clip_big_g_normalized_mse"].detach(),
                        context,
                    ),
                    "clip_big_g_cosine": scalar_mean(
                        component_losses["clip_big_g_cosine"].detach(),
                        context,
                    ),
                    "pooled_mse": scalar_mean(component_losses["pooled_mse"].detach(), context),
                    "pooled_cosine": scalar_mean(
                        component_losses["pooled_cosine"].detach(),
                        context,
                    ),
                    "geometry": scalar_mean(
                        (
                            component_losses["clip_l_geometry"]
                            + component_losses["clip_big_g_geometry"]
                            + component_losses["pooled_geometry"]
                        ).detach()
                        / 3.0,
                        context,
                    ),
                    "retrieval": scalar_mean(
                        (
                            component_losses["clip_l_retrieval"]
                            + component_losses["clip_big_g_retrieval"]
                            + component_losses["pooled_retrieval"]
                        ).detach()
                        / 3.0,
                        context,
                    ),
                    "variance": scalar_mean(
                        (
                            component_losses["clip_l_variance"]
                            + component_losses["clip_big_g_variance"]
                            + component_losses["pooled_variance"]
                        ).detach()
                        / 3.0,
                        context,
                    ),
                    "content_cosine": scalar_mean(
                        (
                            component_losses["clip_l_content_cosine"]
                            + component_losses["clip_big_g_content_cosine"]
                        ).detach()
                        / 2.0,
                        context,
                    ),
                    "padding_cosine": scalar_mean(
                        (
                            component_losses["clip_l_padding_cosine"]
                            + component_losses["clip_big_g_padding_cosine"]
                        ).detach()
                        / 2.0,
                        context,
                    ),
                    "functional_mse": scalar_mean(functional.mse.detach(), context),
                    "functional_cosine": scalar_mean(functional.cosine.detach(), context),
                    "functional_scale": functional_scale,
                    "functional_timestep": functional.timestep,
                    "rollout_mse": scalar_mean(rollout.prediction_mse.detach(), context),
                    "rollout_cosine": scalar_mean(rollout.prediction_cosine.detach(), context),
                    "rollout_transition": scalar_mean(rollout.transition_mse.detach(), context),
                    "rollout_scale": rollout_scale,
                    "rollout_calls": rollout.calls,
                    "closed_loop_prediction_relative_mse": scalar_mean(
                        closed_loop.prediction_relative_mse.detach(),
                        context,
                    ),
                    "closed_loop_transition_relative_mse": scalar_mean(
                        closed_loop.transition_relative_mse.detach(),
                        context,
                    ),
                    "closed_loop_terminal_relative_mse": scalar_mean(
                        closed_loop.terminal_relative_mse.detach(),
                        context,
                    ),
                    "closed_loop_transition_cosine": scalar_mean(
                        closed_loop.transition_cosine.detach(),
                        context,
                    ),
                    "closed_loop_per_step_transition_relative_mse": [
                        scalar_mean(value.detach(), context)
                        for value in closed_loop.per_step_transition_relative_mse
                    ],
                    "closed_loop_scale": closed_loop_scale,
                    "closed_loop_calls": closed_loop.calls,
                    "lr": optimizer.param_groups[0]["lr"],
                    "mode": mode,
                }
                if context.is_main:
                    history.append(item)
                    progress.set_postfix(
                        mode=mode,
                        loss=f"{item['loss']:.4f}",
                        repr=f"{item['representation']:.4f}",
                        func=f"{item['functional_mse']:.4f}",
                        traj=f"{item['closed_loop_terminal_relative_mse']:.4f}",
                    )

            save_latest = current_step % args.save_latest_every == 0
            save_archive = current_step % args.save_archive_every == 0
            is_final = current_step == args.target_step
            if save_latest or save_archive or is_final:
                barrier()
                if context.is_main:
                    payload = {
                        "step": current_step,
                        "image_bridge": unwrap(bridge_model).trainable_state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "args": vars(args),
                        "history": history,
                        "target": "ssd1b_native_dual_clip_condition",
                        "architecture": {
                            "source": "frozen_smolvlm2_single_forward",
                            "token_queries": config.image_bridge.sequence_length,
                            "global_queries": 1,
                            "clip_l_dim": config.image_bridge.clip_l_dim,
                            "clip_big_g_dim": config.image_bridge.clip_big_g_dim,
                            "pooled_dim": config.image_bridge.pooled_dim,
                            "trainable_master_dtype": "float32",
                        },
                        "distillation": {
                            "representation": True,
                            "frozen_unet_functional": True,
                            "objective_version": args.objective_version,
                            "mask_aware_tokens": args.objective_version == "v2",
                            "global_retrieval": args.objective_version == "v2",
                            "variance_alignment": args.objective_version == "v2",
                            "balanced_functional_timesteps": args.objective_version == "v2",
                            "full_lcm_rollout_calls": 4,
                            "teacher_state_policy": (
                                "independent_native_closed_loop"
                                if args.objective_version == "v2"
                                else "detached_student_on_policy"
                            ),
                        },
                    }
                    atomic_torch_save(payload, latest_path)
                    if save_archive or is_final:
                        atomic_torch_save(
                            payload,
                            output_dir / f"ssd1b_image_bridge_step{current_step:06d}.pt",
                        )
                    (output_dir / "history.json").write_text(
                        json.dumps(history, indent=2),
                        encoding="utf-8",
                    )
                    rank0_print(
                        context,
                        f"Saved SSD1B Image Bridge step={current_step} "
                        f"latest=true archive={save_archive or is_final}",
                    )
                barrier()

            progress.update(1)
            if current_step >= args.target_step:
                break
        epoch += 1
        skip_batch = 0

    progress.close()
    rank0_print(context, f"Training complete: {latest_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
