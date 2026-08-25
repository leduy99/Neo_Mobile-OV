#!/usr/bin/env python
# ruff: noqa: E402
"""Reproduce NeoDragon Pyramidal DMD from a monolithic teacher to 1-1-1 DiT.

This is intentionally independent from the Mobile-OV bridges and prior joint
fine-tuning experiments.  It trains only three native NeoDragon components:

* frozen released multi-step DiT teacher with CFG;
* one-step conditional student DiT; and
* one-step conditional fake DiT that tracks the student's endpoint density.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    cleanup_distributed,
    rank0_print,
    scalar_mean,
    setup_distributed,
)
from new_mobile_ov.training.neodragon_pyramidal_dmd import (
    DMDCondition,
    build_stage_pair,
    cauchy_endpoint_loss,
    dmd_sample_weight,
    dmd_surrogate_loss,
    linear_weight_decay,
    motion_residual_anchor_loss,
    predict_flow,
    rollout_history_probability,
    stage_noisy_student_endpoint,
    stage_timestep,
    student_probe_sigmas,
)
from new_mobile_ov.training.neodragon_rollout import (
    downsample_noise_2x,
    prepare_past_conditions,
    pyramid_latents,
    upsample_pyramidal_latent,
)


ALL_NATIVE_SCHEDULE = "pyramidal_1-1-1_all_native_units"
LEGACY_VIDEO_ONLY_SCHEDULE = "hybrid_1-1-1_video_units_only"
ANCHOR_ALT_SCHEDULE = "pyramidal_1-1-1_external_anchor_video_units"
ROLLOUT_AWARE_V3_SCHEDULE = "pyramidal_1-1-1_external_anchor_rollout_aware_video_units"


def protocol_metadata(
    *,
    include_first_unit: bool,
    external_anchor_alternative: bool,
    rollout_aware_v3: bool = False,
) -> tuple[str, str]:
    """Return an unambiguous schedule/objective pair for a DMD run."""

    if include_first_unit and external_anchor_alternative:
        raise ValueError(
            "--external-anchor-alternative requires --no-include-first-unit"
        )
    if rollout_aware_v3 and (include_first_unit or not external_anchor_alternative):
        raise ValueError(
            "--rollout-aware-v3 requires --no-include-first-unit and "
            "--external-anchor-alternative"
        )
    if rollout_aware_v3:
        return (
            ROLLOUT_AWARE_V3_SCHEDULE,
            "pyramidal_dmd_v3_rollout_aware_external_anchor_video_units",
        )
    if include_first_unit:
        return ALL_NATIVE_SCHEDULE, "pyramidal_dmd_reproduction_v2_all_native_units"
    if external_anchor_alternative:
        return ANCHOR_ALT_SCHEDULE, "pyramidal_dmd_v2_alt_external_anchor_video_units"
    return LEGACY_VIDEO_ONLY_SCHEDULE, "pyramidal_dmd_reproduction_v1_legacy_video_units_only"


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def autocast_trainable(device: torch.device, forward_dtype: torch.dtype):
    """Run FP32-master student/fake modules with the requested forward dtype."""

    return torch.autocast(
        device_type=device.type,
        dtype=forward_dtype,
        enabled=device.type == "cuda" and forward_dtype != torch.float32,
    )


def assert_fp32_trainable_parameters(module: torch.nn.Module, *, name: str) -> None:
    mismatched = [
        (parameter_name, parameter.dtype)
        for parameter_name, parameter in unwrap(module).named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if mismatched:
        preview = ", ".join(f"{key}={dtype}" for key, dtype in mismatched[:5])
        raise RuntimeError(
            f"{name} must use FP32 master parameters for DMD optimization; got {preview}"
        )


def resolve_optimizer_backend(requested: str, *, world_size: int) -> str:
    """Select replicated AdamW or ZeRO-1 optimizer-state sharding."""

    requested = str(requested).lower()
    if requested not in {"auto", "adamw", "zero1"}:
        raise ValueError(f"Unsupported optimizer backend: {requested!r}")
    if requested == "auto":
        return "zero1" if world_size > 1 else "adamw"
    if requested == "zero1" and world_size <= 1:
        raise ValueError("ZeRO-1 optimizer sharding requires world_size > 1")
    return requested


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    backend: str,
) -> torch.optim.Optimizer:
    parameters = list(parameters)
    defaults = {
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        # Foreach holds additional tensor lists during the update. The scalar
        # path is slower but keeps the 40 GB A100 peak predictable.
        "foreach": False,
    }
    if backend == "adamw":
        return torch.optim.AdamW(parameters, **defaults)
    if backend == "zero1":
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            parameters,
            optimizer_class=torch.optim.AdamW,
            **defaults,
        )
    raise ValueError(f"Unsupported optimizer backend: {backend!r}")


def consolidate_optimizer_state(optimizer: torch.optim.Optimizer, *, destination_rank: int) -> None:
    """Gather a ZeRO-1 optimizer state before writing a resumable checkpoint."""

    consolidate = getattr(optimizer, "consolidate_state_dict", None)
    if callable(consolidate):
        consolidate(to=destination_rank)


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def select_native_unit_stage(
    step: int,
    *,
    include_first_unit: bool,
    num_stages: int = 3,
) -> tuple[int, int]:
    """Cycle one shared native ``unit x stage`` position across DDP ranks.

    Synthetic trajectories contain the complete native seven-unit T2V video.
    The paper's step-distilled settings likewise generate all seven units with
    the same schedule, so the default includes unit zero.  ``False`` remains
    available only for reproducing the old external-anchor-only ablation.
    """

    if step < 1 or num_stages < 1:
        raise ValueError("step and num_stages must be positive")
    units = 7 if include_first_unit else 6
    position = (int(step) - 1) % (units * int(num_stages))
    relative_unit, stage = divmod(position, int(num_stages))
    return (relative_unit if include_first_unit else relative_unit + 1), stage


@torch.no_grad()
def rollout_student_state_to_position(
    *,
    student: torch.nn.Module,
    scheduler,
    anchor: torch.Tensor,
    full_noise: torch.Tensor,
    condition: DMDCondition,
    target_unit: int,
    target_stage: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
    """Reach one deployed video-unit call using only prior student outputs."""

    if anchor.shape[2] != 1 or full_noise.shape[2] != 7:
        raise ValueError("V3 expects one anchor and seven units of rollout noise")
    if not 1 <= target_unit <= 6 or not 0 <= target_stage < 3:
        raise ValueError("V3 target must be unit 1..6 and stage 0..2")
    low_noise = downsample_noise_2x(full_noise, 2)
    generated = [anchor.detach()]
    calls = 0
    for unit in range(1, target_unit + 1):
        histories = prepare_past_conditions(generated, num_stages=3)
        current = low_noise[:, :, unit : unit + 1]
        for stage in range(3):
            if stage > 0:
                current = upsample_pyramidal_latent(
                    current,
                    orig_sigma=1.0 - scheduler.orig_start_sigmas[stage],
                    gamma=scheduler.config.gamma,
                    generator=generator,
                )
            history = tuple(histories[stage])
            if unit == target_unit and stage == target_stage:
                return current.detach(), tuple(value.detach() for value in history), calls
            timestep = scheduler.get_stage_timesteps(1, stage, device=current.device)[0]
            current = current - predict_flow(
                dit=student,
                current=current,
                history=history,
                condition=condition,
                timestep=timestep,
            )
            current = current.detach()
            calls += 1
        generated.append(current)
    raise RuntimeError("Failed to reach the requested V3 rollout position")


class SyntheticTeacherLatentDataset(Dataset):
    """Random-access synthetic teacher samples written by the preparation job."""

    def __init__(self, manifest: str | Path, *, max_samples: int) -> None:
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if max_samples > 0:
            rows = rows[: int(max_samples)]
        if not rows:
            raise ValueError(f"No DMD synthetic rows found in {self.manifest}")
        required = {"latent_path", "condition_prompt"}
        if not required.issubset(rows[0]):
            raise ValueError(f"{self.manifest} must contain columns {sorted(required)}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = Path(row["latent_path"])
        if not path.is_absolute():
            path = self.root / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload.get("latent") if isinstance(payload, dict) else payload
        if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or latent.shape[1] != 7:
            raise RuntimeError(f"Invalid DMD latent in {path}: {getattr(latent, 'shape', None)}")
        return {"latent": latent.contiguous(), "prompt": str(row["condition_prompt"])}


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "latent": torch.stack([item["latent"] for item in batch], dim=0),
        "prompt": [str(item["prompt"]) for item in batch],
    }


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield epoch, batch
        epoch += 1


def native_condition(
    *,
    text_bundle: torch.nn.Module,
    context_adapter: torch.nn.Module,
    prompts: list[str],
    negative_prompt: str,
    device: torch.device,
    guidance_scale: float,
) -> DMDCondition:
    tokens, mask, pooled = text_bundle(prompts, device)
    tokens = context_adapter(tokens)
    if guidance_scale <= 0:
        return DMDCondition(tokens=tokens, mask=mask, pooled=pooled)
    negatives = [negative_prompt] * len(prompts)
    negative_tokens, negative_mask, negative_pooled = text_bundle(negatives, device)
    negative_tokens = context_adapter(negative_tokens)
    return DMDCondition(
        tokens=tokens,
        mask=mask,
        pooled=pooled,
        negative_tokens=negative_tokens,
        negative_mask=negative_mask,
        negative_pooled=negative_pooled,
        guidance_scale=float(guidance_scale),
    )


def load_models(
    cfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trainable_dtype: torch.dtype,
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

    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_NEGATIVE_PROMPT

    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}", torch_dtype=dtype
    ).to(device).eval()
    teacher = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
    ).to(device).eval()
    student = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=trainable_dtype
    ).to(device).train()
    fake = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=trainable_dtype
    ).to(device).train()
    for module in (text_bundle, context_adapter, teacher):
        module.requires_grad_(False)
    return {
        "text": text_bundle,
        "adapter": context_adapter,
        "teacher": teacher,
        "student": student,
        "fake": fake,
        "scheduler": PyramidFlowMatchEulerDiscreteScheduler(),
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "local_model_path": str(local_model_path),
        "context_adapter_id": MULTISTEP_CONTEXT_ADAPTER_ID,
        "dit_id": MULTISTEP_DIT_ID,
    }


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    student: torch.nn.Module,
    fake: torch.nn.Module,
    student_optimizer: torch.optim.Optimizer,
    fake_optimizer: torch.optim.Optimizer,
    history: list[dict[str, object]],
    args: argparse.Namespace,
    models: dict[str, object],
    archive: bool,
    ctx,
) -> None:
    # ZeRO-1 consolidation is collective, so every rank must enter it even
    # though only rank zero writes the resulting checkpoint.
    if args.save_resume:
        consolidate_optimizer_state(student_optimizer, destination_rank=0)
        consolidate_optimizer_state(fake_optimizer, destination_rank=0)
    if ctx.is_main:
        schedule, objective_name = protocol_metadata(
            include_first_unit=args.include_first_unit,
            external_anchor_alternative=args.external_anchor_alternative,
            rollout_aware_v3=args.rollout_aware_v3,
        )
        base = {
            "step": int(step),
            "student": {key: value.detach().cpu() for key, value in unwrap(student).state_dict().items()},
            "model_type": "neodragon_multistep_dit_distilled_to_conditional_one_step",
            "teacher_dit_id": models["dit_id"],
            "context_adapter_id": models["context_adapter_id"],
            "schedule": schedule,
            "objective": {
                "name": objective_name,
                "teacher": "released_neodragon_multistep_cfg",
                "student_init": "released_neodragon_multistep",
                "fake_init": "released_neodragon_multistep",
                "student_fake_update_ratio": f"1:{args.fake_updates}",
                "student_probe_sigmas": list(student_probe_sigmas()),
                "native_unit_indices": list(range(7)) if args.include_first_unit else list(range(1, 7)),
                "teacher_guidance": {
                    "first_unit": args.teacher_first_guidance,
                    "video_units": args.teacher_video_guidance,
                },
                "dmd_weight": args.dmd_weight,
                "cauchy_weight": args.cauchy_weight,
                "cauchy_final_weight": args.cauchy_final_weight,
                "cauchy_decay_steps": args.cauchy_decay_steps,
                "motion_residual_weight": args.motion_residual_weight,
                "forward_dtype": args.dtype,
                "trainable_master_dtype": args.trainable_dtype,
                "unit_zero_policy": (
                    "optimized_as_native_t2v_unit"
                    if args.include_first_unit
                    else "excluded_from_optimization_used_as_teacher_forced_history_anchor"
                ),
                "deployment_first_frame": (
                    "native_one_step_unit_zero"
                    if args.include_first_unit
                    else "external_ssd1b_dreamlite_or_source_image"
                ),
                "training_history_source": (
                    "teacher_to_deployed_student_rollout_curriculum"
                    if args.rollout_aware_v3
                    else "stored_multistep_teacher_trajectory"
                ),
                "student_history_curriculum": (
                    {
                        "warmup_steps": args.history_warmup_steps,
                        "midpoint_step": args.history_midpoint_step,
                        "final_step": args.history_final_step,
                        "midpoint_probability": args.history_midpoint_probability,
                        "final_probability": args.history_final_probability,
                    }
                    if args.rollout_aware_v3
                    else None
                ),
            },
            "args": vars(args),
            "history": history,
        }
        atomic_save(base, output_dir / "neodragon_pyramidal_dmd_student_latest.pt")
        if args.save_resume:
            resume = dict(base)
            resume["fake"] = {
                key: value.detach().cpu() for key, value in unwrap(fake).state_dict().items()
            }
            resume["student_optimizer"] = student_optimizer.state_dict()
            resume["fake_optimizer"] = fake_optimizer.state_dict()
            atomic_save(resume, output_dir / "neodragon_pyramidal_dmd_resume.pt")
        if archive:
            atomic_save(base, output_dir / f"neodragon_pyramidal_dmd_student_step{step:06d}.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        rank0_print(
            ctx,
            f"Saved NeoDragon Pyramidal-DMD checkpoint step={step} archive={archive}",
        )
    barrier()


def load_resume(
    path: Path,
    *,
    student: torch.nn.Module,
    fake: torch.nn.Module,
    student_optimizer: torch.optim.Optimizer,
    fake_optimizer: torch.optim.Optimizer,
    include_first_unit: bool,
    external_anchor_alternative: bool,
    rollout_aware_v3: bool,
) -> tuple[int, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "student" not in payload or "fake" not in payload:
        raise ValueError(f"{path} is not a resumable Pyramidal-DMD checkpoint")
    expected_schedule, _ = protocol_metadata(
        include_first_unit=include_first_unit,
        external_anchor_alternative=external_anchor_alternative,
        rollout_aware_v3=rollout_aware_v3,
    )
    actual_schedule = payload.get("schedule")
    if actual_schedule != expected_schedule:
        raise ValueError(
            "Refusing to resume a checkpoint with a different unit protocol: "
            f"checkpoint={actual_schedule!r}, requested={expected_schedule!r}."
        )
    unwrap(student).load_state_dict(payload["student"], strict=True)
    unwrap(fake).load_state_dict(payload["fake"], strict=True)
    if "student_optimizer" in payload and "fake_optimizer" in payload:
        student_optimizer.load_state_dict(payload["student_optimizer"])
        fake_optimizer.load_state_dict(payload["fake_optimizer"])
    return int(payload.get("step", 0)), list(payload.get("history", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="output/neodragon_pyramidal_dmd_reproduction")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--student-lr", type=float, default=1e-6)
    parser.add_argument("--fake-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fake-updates", type=int, default=2)
    parser.add_argument("--dmd-weight", type=float, default=1.0)
    parser.add_argument("--cauchy-weight", type=float, default=0.5)
    parser.add_argument("--max-dmd-weight", type=float, default=100.0)
    parser.add_argument(
        "--include-first-unit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Distil all seven native T2V units.  This is the paper-aligned "
            "default for synthetic [7,C,H,W] teacher trajectories."
        ),
    )
    parser.add_argument(
        "--external-anchor-alternative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Label the six-video-unit run as the controlled DMD-v2 external-anchor "
            "alternative. Unit zero from each stored teacher trajectory is used only "
            "as causal history; deployment supplies an external first frame."
        ),
    )
    parser.add_argument(
        "--rollout-aware-v3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the external-anchor V3 curriculum that gradually replaces stored "
            "teacher history and current states with deployed one-step student rollout states."
        ),
    )
    parser.add_argument("--history-warmup-steps", type=int, default=1000)
    parser.add_argument("--history-midpoint-step", type=int, default=4000)
    parser.add_argument("--history-final-step", type=int, default=10000)
    parser.add_argument("--history-midpoint-probability", type=float, default=0.5)
    parser.add_argument("--history-final-probability", type=float, default=0.75)
    parser.add_argument("--cauchy-final-weight", type=float, default=0.1)
    parser.add_argument("--cauchy-decay-steps", type=int, default=4000)
    parser.add_argument("--motion-residual-weight", type=float, default=0.05)
    parser.add_argument("--teacher-first-guidance", type=float, default=7.0)
    parser.add_argument("--teacher-video-guidance", type=float, default=5.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--archive-every", type=int, default=5000)
    parser.add_argument(
        "--save-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save fake model and optimizer states in the large resumable checkpoint.",
    )
    parser.add_argument(
        "--archive-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Archive the final student even when it is off the archive interval.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", default="auto", help="auto, none, or a resumable checkpoint path")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--trainable-dtype",
        default="fp32",
        help=(
            "Master parameter dtype for student and fake. DMD requires fp32 so "
            "AdamW updates are not rounded away; --dtype still controls autocast."
        ),
    )
    parser.add_argument(
        "--optimizer-state-sharding",
        choices=("auto", "adamw", "zero1"),
        default="auto",
        help=(
            "Shard AdamW state across distributed ranks with ZeRO-1. 'auto' uses "
            "ZeRO-1 for world_size > 1 and regular AdamW for single-GPU smoke tests."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.fake_updates < 1:
        raise ValueError("steps, batch-size, and fake-updates must be positive")
    schedule, objective_name = protocol_metadata(
        include_first_unit=args.include_first_unit,
        external_anchor_alternative=args.external_anchor_alternative,
        rollout_aware_v3=args.rollout_aware_v3,
    )
    if args.rollout_aware_v3:
        rollout_history_probability(
            1,
            warmup_steps=args.history_warmup_steps,
            midpoint_step=args.history_midpoint_step,
            final_step=args.history_final_step,
            midpoint_probability=args.history_midpoint_probability,
            final_probability=args.history_final_probability,
        )
        linear_weight_decay(
            1,
            initial=args.cauchy_weight,
            final=args.cauchy_final_weight,
            decay_steps=args.cauchy_decay_steps,
        )
    ctx = setup_distributed()
    try:
        torch.manual_seed(args.seed + ctx.rank)
        random.seed(args.seed + ctx.rank)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + ctx.rank)
            torch.backends.cuda.matmul.allow_tf32 = True

        cfg = load_config(args.config)
        dtype = dtype_from_name(args.dtype)
        trainable_dtype = dtype_from_name(args.trainable_dtype)
        if trainable_dtype != torch.float32:
            raise ValueError(
                "DMD student/fake master parameters must be fp32. "
                "Use --dtype bf16 for autocast, not BF16 trainable weights."
            )
        output_dir = Path(args.output_dir)
        if ctx.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        dataset = SyntheticTeacherLatentDataset(args.manifest, max_samples=args.max_samples)
        sampler = (
            DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed)
            if ctx.is_distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=ctx.device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        )
        if not len(loader):
            raise ValueError("DMD loader has no complete batches; reduce batch size or add data")

        models = load_models(
            cfg,
            device=ctx.device,
            dtype=dtype,
            trainable_dtype=trainable_dtype,
        )
        teacher = models["teacher"]
        student = models["student"]
        fake = models["fake"]
        if ctx.is_distributed:
            student = DDP(
                student,
                device_ids=[ctx.local_rank],
                output_device=ctx.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            fake = DDP(
                fake,
                device_ids=[ctx.local_rank],
                output_device=ctx.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        assert_fp32_trainable_parameters(student, name="student")
        assert_fp32_trainable_parameters(fake, name="fake")

        optimizer_backend = resolve_optimizer_backend(
            args.optimizer_state_sharding,
            world_size=ctx.world_size,
        )
        student_optimizer = build_optimizer(
            student.parameters(),
            lr=args.student_lr,
            weight_decay=args.weight_decay,
            backend=optimizer_backend,
        )
        fake_optimizer = build_optimizer(
            fake.parameters(),
            lr=args.fake_lr,
            weight_decay=args.weight_decay,
            backend=optimizer_backend,
        )
        start_step = 0
        history: list[dict[str, object]] = []
        resume_path = output_dir / "neodragon_pyramidal_dmd_resume.pt"
        if args.resume != "none":
            requested = resume_path if args.resume == "auto" else Path(args.resume)
            if requested.is_file():
                start_step, history = load_resume(
                    requested,
                    student=student,
                    fake=fake,
                    student_optimizer=student_optimizer,
                    fake_optimizer=fake_optimizer,
                    include_first_unit=args.include_first_unit,
                    external_anchor_alternative=args.external_anchor_alternative,
                    rollout_aware_v3=args.rollout_aware_v3,
                )
                rank0_print(ctx, f"Resumed Pyramidal-DMD at step={start_step} from {requested}")

        parameters = sum(parameter.numel() for parameter in unwrap(student).parameters())
        rank0_print(
            ctx,
            "NeoDragon Pyramidal-DMD reproduction: "
            f"world_size={ctx.world_size} batch_per_gpu={args.batch_size} "
            f"global_batch={ctx.world_size * args.batch_size} synthetic_rows={len(dataset)} "
            f"steps={start_step}->{args.steps} student_params={parameters:,} "
            f"student:fake=1:{args.fake_updates} units="
            f"{'0-6' if args.include_first_unit else '1-6'} dtype={dtype} "
            f"trainable_dtype={trainable_dtype} "
            f"optimizer={optimizer_backend} "
            f"schedule={schedule} objective={objective_name}",
        )

        scheduler = models["scheduler"]
        iterator = cycle_loader(loader, sampler)
        rollout_generator = torch.Generator(device=ctx.device)
        rollout_generator.manual_seed(args.seed + 100_003 + ctx.rank)
        progress = tqdm(
            range(start_step, args.steps),
            disable=not ctx.is_main,
            desc="Train NeoDragon Pyramidal-DMD",
        )
        for zero_based_step in progress:
            step = zero_based_step + 1
            _, batch = next(iterator)
            clean_video = batch["latent"].to(ctx.device, dtype=dtype, non_blocking=True)
            prompts = list(batch["prompt"])
            if clean_video.shape[2] != 7:
                raise RuntimeError(f"Expected [B,C,7,H,W] teacher latents, got {tuple(clean_video.shape)}")

            # Every rank takes the same native position. External-anchor
            # protocols reserve stored unit zero as causal first-frame history.
            unit, stage = select_native_unit_stage(
                step,
                include_first_unit=args.include_first_unit,
            )
            video_pyramid = pyramid_latents(clean_video, num_stages=3)
            clean_endpoint = video_pyramid[stage][:, :, unit : unit + 1]
            history_frames = [clean_video[:, :, index : index + 1] for index in range(unit)]
            teacher_past_conditions = tuple(
                prepare_past_conditions(history_frames, num_stages=3)[stage]
            )
            pair = build_stage_pair(
                clean=clean_endpoint,
                scheduler=scheduler,
                stage=stage,
                noise=torch.randn_like(clean_endpoint),
            )
            start_timestep = stage_timestep(
                scheduler, stage=stage, local_sigma=1.0, device=ctx.device
            )

            # Match the released teacher's first/video CFG convention.  The
            # student and fake remain conditional-only, as in Pyramidal DMD.
            teacher_guidance = (
                args.teacher_first_guidance if unit == 0 else args.teacher_video_guidance
            )
            with torch.no_grad():
                teacher_condition = native_condition(
                    text_bundle=models["text"],
                    context_adapter=models["adapter"],
                    prompts=prompts,
                    negative_prompt=str(models["negative_prompt"]),
                    device=ctx.device,
                    guidance_scale=teacher_guidance,
                )
                student_condition = native_condition(
                    text_bundle=models["text"],
                    context_adapter=models["adapter"],
                    prompts=prompts,
                    negative_prompt=str(models["negative_prompt"]),
                    device=ctx.device,
                    guidance_scale=0.0,
                )

            student_history_probability = 0.0
            use_student_history = False
            rollout_calls = 0
            past_conditions = teacher_past_conditions
            stage_start = pair.start
            if args.rollout_aware_v3:
                student_history_probability = rollout_history_probability(
                    step,
                    warmup_steps=args.history_warmup_steps,
                    midpoint_step=args.history_midpoint_step,
                    final_step=args.history_final_step,
                    midpoint_probability=args.history_midpoint_probability,
                    final_probability=args.history_final_probability,
                )
                decision_rng = random.Random(args.seed + 1_000_003 * step)
                use_student_history = (
                    decision_rng.random() < student_history_probability
                )
            if use_student_history:
                full_noise = torch.randn(
                    clean_video.shape,
                    device=ctx.device,
                    dtype=dtype,
                    generator=rollout_generator,
                )
                with autocast_trainable(ctx.device, dtype):
                    stage_start, past_conditions, rollout_calls = (
                        rollout_student_state_to_position(
                            student=student,
                            scheduler=scheduler,
                            anchor=clean_video[:, :, :1],
                            full_noise=full_noise,
                            condition=student_condition,
                            target_unit=unit,
                            target_stage=stage,
                            generator=rollout_generator,
                        )
                    )

            # The fake model tracks the moving endpoint distribution. Its target
            # is Pyramidal Flow Matching on detached current student endpoints.
            set_requires_grad(unwrap(student), False)
            set_requires_grad(unwrap(fake), True)
            fake_losses: list[torch.Tensor] = []
            with torch.no_grad():
                with autocast_trainable(ctx.device, dtype):
                    student_at_start = predict_flow(
                        dit=student,
                        current=stage_start,
                        history=past_conditions,
                        condition=student_condition,
                        timestep=start_timestep,
                    )
                detached_endpoint = (stage_start - student_at_start).detach()
            for _ in range(args.fake_updates):
                fake_optimizer.zero_grad(set_to_none=True)
                fake_tau = torch.rand((), device=ctx.device).item()
                fake_noise = torch.randn_like(detached_endpoint)
                fake_probe, fake_start, fake_end = stage_noisy_student_endpoint(
                    endpoint=detached_endpoint,
                    scheduler=scheduler,
                    stage=stage,
                    local_sigma=fake_tau,
                    noise=fake_noise,
                )
                fake_timestep = stage_timestep(
                    scheduler, stage=stage, local_sigma=fake_tau, device=ctx.device
                )
                with autocast_trainable(ctx.device, dtype):
                    fake_prediction = predict_flow(
                        dit=fake,
                        current=fake_probe,
                        history=past_conditions,
                        condition=student_condition,
                        timestep=fake_timestep,
                    )
                fake_loss = torch.nn.functional.mse_loss(
                    fake_prediction.float(), (fake_start - fake_end).float()
                )
                fake_loss.backward()
                torch.nn.utils.clip_grad_norm_(unwrap(fake).parameters(), args.clip_grad_norm)
                fake_optimizer.step()
                fake_losses.append(fake_loss.detach())
                fake_optimizer.zero_grad(set_to_none=True)

            # The student sees one conditional model call. Teacher and fake are
            # evaluated at the same re-noised endpoint and contribute only their
            # DMD direction, never gradients through their parameters.
            set_requires_grad(unwrap(student), True)
            set_requires_grad(unwrap(fake), False)
            student_optimizer.zero_grad(set_to_none=True)
            with autocast_trainable(ctx.device, dtype):
                student_prediction = predict_flow(
                    dit=student,
                    current=stage_start,
                    history=past_conditions,
                    condition=student_condition,
                    timestep=start_timestep,
                )
            student_endpoint = stage_start - student_prediction
            # Rotate the four fixed student probes after a complete shared
            # unit/stage pass.  The corrected native-T2V protocol has 7 x 3
            # positions; the legacy external-anchor ablation has 6 x 3.
            positions_per_pass = (7 if args.include_first_unit else 6) * 3
            tau = student_probe_sigmas()[
                ((step - 1) // positions_per_pass) % len(student_probe_sigmas())
            ]
            probe_noise = torch.randn_like(student_endpoint)
            probe, probe_start, probe_end = stage_noisy_student_endpoint(
                endpoint=student_endpoint,
                scheduler=scheduler,
                stage=stage,
                local_sigma=tau,
                noise=probe_noise,
            )
            probe_timestep = stage_timestep(scheduler, stage=stage, local_sigma=tau, device=ctx.device)
            with torch.no_grad():
                with autocast_trainable(ctx.device, dtype):
                    teacher_flow = predict_flow(
                        dit=teacher,
                        current=probe.detach(),
                        history=tuple(value.detach() for value in past_conditions),
                        condition=teacher_condition,
                        timestep=probe_timestep,
                    )
                    fake_flow = predict_flow(
                        dit=fake,
                        current=probe.detach(),
                        history=tuple(value.detach() for value in past_conditions),
                        condition=student_condition,
                        timestep=probe_timestep,
                    )
            weights = dmd_sample_weight(
                teacher_flow=teacher_flow,
                stage_flow_target=probe_start - probe_end,
                maximum=args.max_dmd_weight,
            )
            dmd_loss, direction_rms = dmd_surrogate_loss(
                endpoint=student_endpoint,
                teacher_flow=teacher_flow,
                fake_flow=fake_flow,
                sample_weight=weights,
            )
            cauchy_loss = cauchy_endpoint_loss(student_endpoint, clean_endpoint)
            scheduled_cauchy_weight = (
                linear_weight_decay(
                    step,
                    initial=args.cauchy_weight,
                    final=args.cauchy_final_weight,
                    decay_steps=args.cauchy_decay_steps,
                )
                if args.rollout_aware_v3
                else args.cauchy_weight
            )
            effective_cauchy_weight = (
                0.0 if use_student_history else scheduled_cauchy_weight
            )
            motion_loss = student_endpoint.new_zeros((), dtype=torch.float32)
            effective_motion_weight = 0.0
            if args.rollout_aware_v3 and not use_student_history:
                previous_endpoint = video_pyramid[stage][
                    :, :, unit - 1 : unit
                ]
                motion_loss = motion_residual_anchor_loss(
                    student_endpoint,
                    clean_endpoint,
                    previous_endpoint,
                )
                effective_motion_weight = args.motion_residual_weight
            loss = (
                args.dmd_weight * dmd_loss
                + effective_cauchy_weight * cauchy_loss
                + effective_motion_weight * motion_loss
            )
            dmd_endpoint_grad = torch.autograd.grad(
                args.dmd_weight * dmd_loss,
                student_endpoint,
                retain_graph=True,
            )[0]
            auxiliary_loss = (
                effective_cauchy_weight * cauchy_loss
                + effective_motion_weight * motion_loss
            )
            if effective_cauchy_weight > 0.0 or effective_motion_weight > 0.0:
                auxiliary_endpoint_grad = torch.autograd.grad(
                    auxiliary_loss,
                    student_endpoint,
                    retain_graph=True,
                )[0]
            else:
                auxiliary_endpoint_grad = torch.zeros_like(student_endpoint)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap(student).parameters(), args.clip_grad_norm)
            student_optimizer.step()
            student_optimizer.zero_grad(set_to_none=True)
            set_requires_grad(unwrap(fake), True)

            endpoint_mse = torch.nn.functional.mse_loss(student_endpoint.float(), clean_endpoint.float())
            teacher_pf_error = torch.nn.functional.mse_loss(
                teacher_flow.float(), (probe_start - probe_end).float()
            )
            dmd_grad_rms = dmd_endpoint_grad.float().square().mean().sqrt()
            auxiliary_grad_rms = auxiliary_endpoint_grad.float().square().mean().sqrt()
            dmd_aux_grad_ratio = dmd_grad_rms / auxiliary_grad_rms.clamp_min(1e-12)
            history_relative_l2 = clean_endpoint.new_zeros((), dtype=torch.float32)
            if use_student_history:
                history_error = sum(
                    (student_value.float() - teacher_value.float()).square().sum()
                    for student_value, teacher_value in zip(
                        past_conditions, teacher_past_conditions
                    )
                )
                history_reference = sum(
                    teacher_value.float().square().sum()
                    for teacher_value in teacher_past_conditions
                )
                history_relative_l2 = history_error / history_reference.clamp_min(1e-12)
            metrics = {
                "step": step,
                "unit": unit,
                "stage": stage,
                "tau": tau,
                "loss": scalar_mean(loss, ctx),
                "dmd_surrogate": scalar_mean(dmd_loss, ctx),
                "cauchy": scalar_mean(cauchy_loss, ctx),
                "cauchy_weight": float(effective_cauchy_weight),
                "motion_residual": scalar_mean(motion_loss, ctx),
                "motion_weight": float(effective_motion_weight),
                "endpoint_mse": scalar_mean(endpoint_mse, ctx),
                "direction_rms": scalar_mean(direction_rms, ctx),
                "fake_mse": scalar_mean(torch.stack(fake_losses).mean(), ctx),
                "teacher_pf_error": scalar_mean(teacher_pf_error, ctx),
                "mean_sample_weight": scalar_mean(weights.mean(), ctx),
                "history_source": "student_rollout" if use_student_history else "teacher",
                "student_history_probability": float(student_history_probability),
                "history_relative_l2": scalar_mean(history_relative_l2, ctx),
                "rollout_calls": int(rollout_calls),
                "dmd_endpoint_grad_rms": scalar_mean(dmd_grad_rms, ctx),
                "auxiliary_endpoint_grad_rms": scalar_mean(auxiliary_grad_rms, ctx),
                "dmd_aux_grad_ratio": scalar_mean(dmd_aux_grad_ratio, ctx),
            }
            if step % args.log_every == 0 or step == args.steps:
                history.append(metrics)
                progress.set_postfix(
                    unit=unit,
                    stage=stage,
                    cauchy=f"{metrics['cauchy']:.3f}",
                    dmd=f"{metrics['dmd_surrogate']:.3f}",
                    fake=f"{metrics['fake_mse']:.3f}",
                    endpoint=f"{metrics['endpoint_mse']:.3f}",
                    history="student" if use_student_history else "teacher",
                    cweight=f"{effective_cauchy_weight:.2f}",
                )
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    student=student,
                    fake=fake,
                    student_optimizer=student_optimizer,
                    fake_optimizer=fake_optimizer,
                    history=history,
                    args=args,
                    models=models,
                    archive=(
                        step % args.archive_every == 0
                        or (args.archive_final and step == args.steps)
                    ),
                    ctx=ctx,
                )
        rank0_print(ctx, f"Completed NeoDragon Pyramidal-DMD reproduction at step={args.steps}.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
