#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVDreamLiteImageBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.dreamlite_distillation import (
    DreamLiteClosedLoopResult,
    DreamLiteFrozenController,
    DreamLiteFrozenQwenTeacher,
    DreamLiteFunctionalResult,
    DreamLiteResolutionBucket,
    dreamlite_direct_representation_losses,
    dreamlite_representation_losses,
    parse_dreamlite_resolution_buckets,
)
from new_mobile_ov.training.neodragon_objectives import linear_ramp


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
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, low_memory=False)
        fallback = fallback_column if fallback_column in frame.columns else None
        if fallback is None:
            fallback = next((name for name in ("caption", "prompt", "text") if name in frame.columns), None)
        available_variants = [name for name in variant_columns if name in frame.columns]
        if fallback is None and not available_variants:
            raise ValueError(f"{path} has no caption, prompt, or text column.")
        self.items: list[list[tuple[str, float]]] = []
        for _, row in frame.iterrows():
            choices = []
            for name, weight in zip(variant_columns, variant_weights):
                if name in frame.columns:
                    value = clean_text(row.get(name))
                    if value:
                        choices.append((value, weight))
            if not choices and fallback is not None:
                value = clean_text(row.get(fallback))
                if value:
                    choices.append((value, 1.0))
            if choices:
                self.items.append(choices)
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"No valid generation prompts found in {path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> str:
        choices = self.items[index]
        return random.choices(
            [value for value, _ in choices],
            weights=[weight for _, weight in choices],
            k=1,
        )[0]


class EditDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        image_column: str,
        instruction_column: str,
        max_samples: int,
    ) -> None:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            frame = pd.read_json(path, lines=True)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, low_memory=False)
        image_column = image_column if image_column in frame.columns else next(
            (name for name in ("source_image", "image_path", "image") if name in frame.columns),
            "",
        )
        instruction_column = instruction_column if instruction_column in frame.columns else next(
            (name for name in ("instruction", "edit_instruction", "prompt", "text") if name in frame.columns),
            "",
        )
        if not image_column or not instruction_column:
            raise ValueError(
                f"{path} needs source image and edit instruction columns; got {list(frame.columns)}"
            )
        self.items = []
        for _, row in frame.iterrows():
            image_path = Path(clean_text(row.get(image_column))).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            instruction = clean_text(row.get(instruction_column))
            if image_path.is_file() and instruction:
                self.items.append((str(image_path), instruction))
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"No valid edit samples found in {path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.items[index]


def edit_collate(items: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    paths, prompts = zip(*items)
    return list(paths), list(prompts)


def dtype_from_name(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def unwrap(model):
    return getattr(model, "module", model)


def lr_scale(step: int, *, total_steps: int, warmup_steps: int, final_scale: float) -> float:
    if step <= warmup_steps:
        return max(step / float(max(warmup_steps, 1)), 1e-3)
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return final_scale + (1.0 - final_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def representation_total(losses: dict[str, torch.Tensor], args) -> torch.Tensor:
    total = (
        args.token_mse_weight * losses["token_normalized_mse"]
        + args.token_cos_weight * losses["token_cosine"]
        + args.token_norm_weight * losses["token_norm"]
        + args.pooled_mse_weight * losses["pooled_normalized_mse"]
        + args.pooled_cos_weight * losses["pooled_cosine"]
        + args.geometry_weight * losses["geometry"]
        + args.variance_weight * losses["variance"]
    )
    if "token_mean" in losses:
        total = total + args.token_mean_weight * losses["token_mean"]
    if "token_std" in losses:
        total = total + args.token_std_weight * losses["token_std"]
    return total


def projected_representation_total(
    losses: dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    return (
        0.50 * losses["token_normalized_mse"]
        + losses["token_cosine"]
        + args.projected_pooled_cos_weight * losses["pooled_cosine"]
        + args.projected_moment_weight * losses["token_mean"]
        + args.projected_moment_weight * losses["token_std"]
    )


def choose_resolution_bucket(
    buckets: list[DreamLiteResolutionBucket],
    *,
    seed: int,
    step: int,
) -> DreamLiteResolutionBucket:
    # The seed does not contain rank so every DDP process runs the same graph.
    rng = random.Random(seed + 104729 * step)
    return rng.choices(buckets, weights=[bucket.weight for bucket in buckets], k=1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Qwen3-VL into the Mobile-OV DreamLite bridge.")
    parser.add_argument("--config", default="configs/mobile_ov_dreamlite.yaml")
    parser.add_argument("--generation-prompts", required=True)
    parser.add_argument("--edit-manifest", default=None)
    parser.add_argument("--edit-image-column", default="source_image")
    parser.add_argument("--edit-instruction-column", default="instruction")
    parser.add_argument("--edit-probability", type=float, default=0.25)
    parser.add_argument("--output-dir", default="output/dreamlite_image_bridge")
    parser.add_argument("--target-step", type=int, default=100000)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--resolution-buckets",
        default="",
        help=(
            "Comma-separated actual@logical resolution buckets: "
            "WIDTHxHEIGHT@TIME_WIDTHxTIME_HEIGHT:WEIGHT. "
            "An empty value preserves the legacy --width/--height behavior."
        ),
    )
    parser.add_argument("--caption-variant-columns", default="caption_short,caption_medium,caption_long")
    parser.add_argument("--caption-variant-weights", default="1,1,1")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--token-mse-weight", type=float, default=0.50)
    parser.add_argument("--token-cos-weight", type=float, default=1.00)
    parser.add_argument("--token-norm-weight", type=float, default=0.05)
    parser.add_argument("--pooled-mse-weight", type=float, default=0.50)
    parser.add_argument("--pooled-cos-weight", type=float, default=1.00)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument(
        "--representation-mode",
        choices=("interpolated", "direct"),
        default="interpolated",
    )
    parser.add_argument("--token-mean-weight", type=float, default=0.0)
    parser.add_argument("--token-std-weight", type=float, default=0.0)
    parser.add_argument("--projected-weight", type=float, default=0.0)
    parser.add_argument("--projected-pooled-cos-weight", type=float, default=0.5)
    parser.add_argument("--projected-moment-weight", type=float, default=0.25)
    parser.add_argument("--representation-final-scale", type=float, default=0.25)
    parser.add_argument("--functional-weight", type=float, default=5.0)
    parser.add_argument("--functional-cos-weight", type=float, default=0.5)
    parser.add_argument("--functional-start-step", type=int, default=10001)
    parser.add_argument("--functional-ramp-steps", type=int, default=5000)
    parser.add_argument("--functional-batch-size", type=int, default=1)
    parser.add_argument("--closed-loop-weight", type=float, default=2.0)
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
    parser.add_argument("--training-version", default="legacy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = setup_distributed()
    torch.manual_seed(args.seed + context.rank)
    random.seed(args.seed + context.rank)
    config = load_config(args.config)
    inference_dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resolution_buckets:
        resolution_buckets = parse_dreamlite_resolution_buckets(args.resolution_buckets)
    else:
        resolution_buckets = [
            DreamLiteResolutionBucket(
                width=args.width,
                height=args.height,
                time_id_width=args.width,
                time_id_height=args.height,
                weight=1.0,
            )
        ]

    columns = split_csv(args.caption_variant_columns)
    weights = [float(value) for value in split_csv(args.caption_variant_weights)]
    if len(columns) != len(weights):
        raise ValueError("caption variant columns and weights must have the same length")
    generation_data = CaptionDataset(
        args.generation_prompts,
        variant_columns=columns,
        variant_weights=weights,
        fallback_column=args.caption_fallback_column,
        max_samples=args.max_samples,
    )
    generation_sampler = DistributedSampler(
        generation_data,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=args.seed,
    ) if context.is_distributed else None
    generation_loader = DataLoader(
        generation_data,
        batch_size=args.batch_size,
        sampler=generation_sampler,
        shuffle=generation_sampler is None,
        num_workers=0,
        drop_last=True,
    )
    edit_loader = None
    edit_sampler = None
    if args.edit_manifest:
        edit_data = EditDataset(
            args.edit_manifest,
            image_column=args.edit_image_column,
            instruction_column=args.edit_instruction_column,
            max_samples=args.max_samples,
        )
        edit_sampler = DistributedSampler(
            edit_data,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed + 17,
        ) if context.is_distributed else None
        edit_loader = DataLoader(
            edit_data,
            batch_size=args.batch_size,
            sampler=edit_sampler,
            shuffle=edit_sampler is None,
            num_workers=0,
            drop_last=True,
            collate_fn=edit_collate,
        )

    rank0_print(context, "Loading frozen SmolVLM2, Qwen3-VL teacher, and DreamLite controller...")
    bridge = MobileOVDreamLiteImageBridge(
        config.bridge,
        config.dreamlite_bridge,
        device=context.device,
        dtype=inference_dtype,
    )
    bridge.promote_trainable_parameters_to_fp32()
    teacher = DreamLiteFrozenQwenTeacher(
        config.dreamlite,
        device=context.device,
        dtype=inference_dtype,
    )
    controller = DreamLiteFrozenController(
        config.dreamlite,
        device=context.device,
        dtype=inference_dtype,
    )
    trainable = [parameter for parameter in bridge.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    current_step = 0
    resume_path = output_dir / "dreamlite_image_bridge_latest.pt" if args.resume == "auto" else Path(args.resume)
    if args.resume != "none" and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        bridge.load_trainable_state_dict(payload["bridge"])
        optimizer.load_state_dict(payload["optimizer"])
        current_step = int(payload["step"])
        rank0_print(context, f"Resumed DreamLite bridge from {resume_path} at step={current_step}")
    if context.is_distributed:
        ddp_devices = [context.local_rank] if context.device.type == "cuda" else None
        bridge = DDP(bridge, device_ids=ddp_devices, broadcast_buffers=False)

    trainable_count = sum(parameter.numel() for parameter in trainable)
    rank0_print(
        context,
        f"DreamLite bridge distillation: world_size={context.world_size} "
        f"batch_per_gpu={args.batch_size} global_batch={args.batch_size * context.world_size} "
        f"trainable_params={trainable_count:,} generation_rows={len(generation_data)} "
        f"edit_rows={len(edit_loader.dataset) if edit_loader else 0} target_step={args.target_step}",
    )
    rank0_print(
        context,
        "Resolution buckets: "
        + ", ".join(f"{bucket.label}:{bucket.weight:g}" for bucket in resolution_buckets),
    )
    generation_epoch = 0
    edit_epoch = 0
    generation_iter = iter(generation_loader)
    edit_iter = iter(edit_loader) if edit_loader else None
    history_path = output_dir / "history.jsonl"
    progress = tqdm(
        total=max(args.target_step - current_step, 0),
        desc="Train Mobile-OV DreamLite image bridge",
        disable=not context.is_main,
    )

    def next_generation():
        nonlocal generation_iter, generation_epoch
        try:
            return next(generation_iter)
        except StopIteration:
            generation_epoch += 1
            if generation_sampler is not None:
                generation_sampler.set_epoch(generation_epoch)
            generation_iter = iter(generation_loader)
            return next(generation_iter)

    def next_edit():
        nonlocal edit_iter, edit_epoch
        if edit_loader is None or edit_iter is None:
            raise RuntimeError("Edit loader is unavailable")
        try:
            return next(edit_iter)
        except StopIteration:
            edit_epoch += 1
            if edit_sampler is not None:
                edit_sampler.set_epoch(edit_epoch)
            edit_iter = iter(edit_loader)
            return next(edit_iter)

    try:
        while current_step < args.target_step:
            current_step += 1
            scale = lr_scale(
                current_step,
                total_steps=args.target_step,
                warmup_steps=args.lr_warmup_steps,
                final_scale=args.lr_final_scale,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            # Every DDP rank must execute the same objective branch. Samples
            # remain rank-sharded, but mode selection is deterministic globally.
            mode_rng = random.Random(args.seed + current_step)
            resolution = choose_resolution_bucket(
                resolution_buckets,
                seed=args.seed,
                step=current_step,
            )
            functional_call_index = random.Random(
                args.seed + 130363 * current_step
            ).randrange(controller.num_steps)
            use_edit = edit_loader is not None and mode_rng.random() < args.edit_probability
            if use_edit:
                image_paths, prompts = next_edit()
                images = [Image.open(path).convert("RGB") for path in image_paths]
                mode = "edit"
            else:
                prompts = list(next_generation())
                images = None
                mode = "generate"
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = context.device.type == "cuda" and inference_dtype != torch.float32
            with torch.autocast(
                device_type=context.device.type,
                dtype=inference_dtype,
                enabled=autocast_enabled,
            ):
                student = bridge(prompts, mode=mode, images=images)
                teacher_condition = teacher.encode(prompts, mode=mode, images=images)
                use_direct_alignment = args.representation_mode == "direct" and mode == "generate"
                if use_direct_alignment:
                    repr_losses = dreamlite_direct_representation_losses(student, teacher_condition)
                    projected_student = controller.project_condition(student)
                    projected_teacher = controller.project_condition(teacher_condition)
                    projected_losses = dreamlite_direct_representation_losses(
                        projected_student,
                        projected_teacher,
                    )
                    projected_value = projected_representation_total(projected_losses, args)
                else:
                    repr_losses = dreamlite_representation_losses(student, teacher_condition)
                    projected_losses = None
                    projected_value = student.prompt_embeds.new_zeros((), dtype=torch.float32)
                repr_value = representation_total(repr_losses, args)
                repr_value = repr_value + args.projected_weight * projected_value
                if args.closed_loop_weight <= 0 or current_step < args.closed_loop_start_step:
                    repr_scale = 1.0
                else:
                    repr_scale = args.representation_final_scale
                functional = DreamLiteFunctionalResult(
                    relative_mse=repr_value.new_zeros(()),
                    cosine=repr_value.new_zeros(()),
                    call_index=-1,
                )
                closed = DreamLiteClosedLoopResult(
                    prediction_relative_mse=repr_value.new_zeros(()),
                    prediction_cosine=repr_value.new_zeros(()),
                    transition_relative_mse=repr_value.new_zeros(()),
                    transition_cosine=repr_value.new_zeros(()),
                    terminal_relative_mse=repr_value.new_zeros(()),
                    calls=0,
                )
                functional_scale = linear_ramp(
                    current_step,
                    start_step=args.functional_start_step,
                    ramp_steps=args.functional_ramp_steps,
                )
                closed_scale = linear_ramp(
                    current_step,
                    start_step=args.closed_loop_start_step,
                    ramp_steps=args.closed_loop_ramp_steps,
                )
                run_closed = (
                    args.closed_loop_weight > 0
                    and current_step >= args.closed_loop_start_step
                    and (current_step - args.closed_loop_start_step) % args.closed_loop_every == 0
                )
                if run_closed:
                    phase = "closed_loop"
                    closed = controller.closed_loop_loss(
                        student,
                        teacher_condition,
                        source_images=images,
                        height=resolution.height,
                        width=resolution.width,
                        batch_size=args.closed_loop_batch_size,
                    )
                elif current_step >= args.functional_start_step:
                    phase = "functional"
                    functional = controller.functional_loss(
                        student,
                        teacher_condition,
                        source_images=images,
                        height=resolution.height,
                        width=resolution.width,
                        time_id_height=resolution.time_id_height,
                        time_id_width=resolution.time_id_width,
                        batch_size=args.functional_batch_size,
                        call_index=functional_call_index,
                    )
                else:
                    phase = "representation"
                loss = repr_scale * repr_value
                loss = loss + functional_scale * (
                    args.functional_weight * functional.relative_mse
                    + args.functional_cos_weight * functional.cosine
                )
                loss = loss + closed_scale * args.closed_loop_weight * (
                    args.closed_loop_prediction_weight * closed.prediction_relative_mse
                    + args.closed_loop_cos_weight
                    * (closed.prediction_cosine + closed.transition_cosine)
                    + args.closed_loop_transition_weight * closed.transition_relative_mse
                    + args.closed_loop_terminal_weight * closed.terminal_relative_mse
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
            optimizer.step()

            should_log = current_step == 1 or current_step % args.log_every == 0
            if should_log:
                item = {
                    "step": current_step,
                    "mode": mode,
                    "phase": phase,
                    "resolution_bucket": resolution.label,
                    "actual_width": resolution.width,
                    "actual_height": resolution.height,
                    "time_id_width": resolution.time_id_width,
                    "time_id_height": resolution.time_id_height,
                    "loss": scalar_mean(loss.detach(), context),
                    "representation": scalar_mean(repr_value.detach(), context),
                    "projected_representation": scalar_mean(projected_value.detach(), context),
                    "token_cosine": scalar_mean(repr_losses["token_cosine"].detach(), context),
                    "pooled_cosine": scalar_mean(repr_losses["pooled_cosine"].detach(), context),
                    "token_mean": scalar_mean(
                        repr_losses.get("token_mean", repr_value.new_zeros(())).detach(),
                        context,
                    ),
                    "token_std": scalar_mean(
                        repr_losses.get("token_std", repr_value.new_zeros(())).detach(),
                        context,
                    ),
                    "mask_agreement": scalar_mean(
                        repr_losses.get("mask_agreement", repr_value.new_ones(())).detach(),
                        context,
                    ),
                    "functional_relative_mse": scalar_mean(functional.relative_mse.detach(), context),
                    "functional_call_index": functional.call_index,
                    "closed_terminal_relative_mse": scalar_mean(closed.terminal_relative_mse.detach(), context),
                    "grad_norm": scalar_mean(torch.as_tensor(grad_norm, device=context.device), context),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if context.device.type == "cuda":
                    item["cuda_peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated(context.device) / (1024**3)
                    )
                    item["cuda_peak_reserved_gib"] = (
                        torch.cuda.max_memory_reserved(context.device) / (1024**3)
                    )
                if context.is_main:
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item) + "\n")
                    progress.set_postfix(
                        mode=mode,
                        phase=phase,
                        loss=f"{item['loss']:.4f}",
                        func=f"{item['functional_relative_mse']:.4f}",
                        roll=f"{item['closed_terminal_relative_mse']:.4f}",
                        res=resolution.label,
                    )
            progress.update(1)

            save_latest = current_step % args.save_latest_every == 0 or current_step == args.target_step
            save_archive = current_step % args.save_archive_every == 0 or current_step == args.target_step
            if save_latest or save_archive:
                barrier()
                if context.is_main:
                    module = unwrap(bridge)
                    payload = {
                        "step": current_step,
                        "bridge": {key: value.detach().cpu() for key, value in module.trainable_state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "config": vars(args),
                        "architecture": (
                            "MobileOVDreamLiteCompactBridgeV4"
                            if args.training_version.lower() == "v4"
                            else "MobileOVDreamLiteCompactBridgeV3"
                            if args.representation_mode == "direct"
                            else "MobileOVDreamLiteImageBridge"
                        ),
                        "teacher": "Qwen3-VL BF16 from DreamLite-mobile",
                        "functional_teacher": (
                            "frozen DreamLite-mobile UNet, native 4-call schedule, "
                            "same-state teacher-forced prefixes"
                            if args.training_version.lower() == "v4"
                            else "frozen DreamLite-mobile UNet, native 4-call schedule"
                        ),
                    }
                    if save_latest:
                        torch.save(payload, output_dir / "dreamlite_image_bridge_latest.pt")
                    if save_archive:
                        torch.save(
                            payload,
                            output_dir / f"dreamlite_image_bridge_step{current_step:06d}.pt",
                        )
                    rank0_print(
                        context,
                        f"Saved DreamLite bridge step={current_step} latest={save_latest} archive={save_archive}",
                    )
                barrier()
    finally:
        progress.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
