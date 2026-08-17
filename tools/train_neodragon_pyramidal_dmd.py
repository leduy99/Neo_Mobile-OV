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
    predict_flow,
    stage_noisy_student_endpoint,
    stage_timestep,
    student_probe_sigmas,
)
from new_mobile_ov.training.neodragon_rollout import pyramid_latents, prepare_past_conditions


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


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


def load_models(cfg, *, device: torch.device, dtype: torch.dtype):
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
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
    ).to(device).train()
    fake = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{MULTISTEP_DIT_ID}", torch_dtype=dtype
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
    if ctx.is_main:
        base = {
            "step": int(step),
            "student": {key: value.detach().cpu() for key, value in unwrap(student).state_dict().items()},
            "model_type": "neodragon_multistep_dit_distilled_to_conditional_one_step",
            "teacher_dit_id": models["dit_id"],
            "context_adapter_id": models["context_adapter_id"],
            "schedule": (
                "pyramidal_1-1-1_all_native_units"
                if args.include_first_unit
                else "hybrid_1-1-1_video_units_only"
            ),
            "objective": {
                "name": (
                    "pyramidal_dmd_reproduction_v2_all_native_units"
                    if args.include_first_unit
                    else "pyramidal_dmd_reproduction_v1_legacy_video_units_only"
                ),
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
            },
            "args": vars(args),
            "history": history,
        }
        resume = dict(base)
        resume["fake"] = {key: value.detach().cpu() for key, value in unwrap(fake).state_dict().items()}
        resume["student_optimizer"] = student_optimizer.state_dict()
        resume["fake_optimizer"] = fake_optimizer.state_dict()
        atomic_save(base, output_dir / "neodragon_pyramidal_dmd_student_latest.pt")
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
) -> tuple[int, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "student" not in payload or "fake" not in payload:
        raise ValueError(f"{path} is not a resumable Pyramidal-DMD checkpoint")
    expected_schedule = (
        "pyramidal_1-1-1_all_native_units"
        if include_first_unit
        else "hybrid_1-1-1_video_units_only"
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
    parser.add_argument("--teacher-first-guidance", type=float, default=7.0)
    parser.add_argument("--teacher-video-guidance", type=float, default=5.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--archive-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", default="auto", help="auto, none, or a resumable checkpoint path")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.fake_updates < 1:
        raise ValueError("steps, batch-size, and fake-updates must be positive")
    ctx = setup_distributed()
    try:
        torch.manual_seed(args.seed + ctx.rank)
        random.seed(args.seed + ctx.rank)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + ctx.rank)
            torch.backends.cuda.matmul.allow_tf32 = True

        cfg = load_config(args.config)
        dtype = dtype_from_name(args.dtype)
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

        models = load_models(cfg, device=ctx.device, dtype=dtype)
        teacher = models["teacher"]
        student = models["student"]
        fake = models["fake"]
        if ctx.is_distributed:
            student = DDP(student, device_ids=[ctx.local_rank], output_device=ctx.local_rank, broadcast_buffers=False)
            fake = DDP(fake, device_ids=[ctx.local_rank], output_device=ctx.local_rank, broadcast_buffers=False)

        student_optimizer = torch.optim.AdamW(
            student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay
        )
        fake_optimizer = torch.optim.AdamW(fake.parameters(), lr=args.fake_lr, weight_decay=args.weight_decay)
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
            f"{'0-6' if args.include_first_unit else '1-6'} dtype={dtype}",
        )

        scheduler = models["scheduler"]
        iterator = cycle_loader(loader, sampler)
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

            # Every rank takes the same native position.  Unit zero is a real
            # T2V training target here because our synthetic data was created
            # with image=None, not an external first-frame anchor.
            unit, stage = select_native_unit_stage(
                step,
                include_first_unit=args.include_first_unit,
            )
            video_pyramid = pyramid_latents(clean_video, num_stages=3)
            clean_endpoint = video_pyramid[stage][:, :, unit : unit + 1]
            history_frames = [clean_video[:, :, index : index + 1] for index in range(unit)]
            past_conditions = tuple(
                prepare_past_conditions(history_frames, num_stages=3)[stage]
            )
            noise = torch.randn_like(clean_endpoint)
            pair = build_stage_pair(clean=clean_endpoint, scheduler=scheduler, stage=stage, noise=noise)
            start_timestep = stage_timestep(scheduler, stage=stage, local_sigma=1.0, device=ctx.device)

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

            # The fake model tracks the moving endpoint distribution. Its target
            # is Pyramidal Flow Matching on detached current student endpoints.
            set_requires_grad(unwrap(student), False)
            set_requires_grad(unwrap(fake), True)
            fake_losses: list[torch.Tensor] = []
            with torch.no_grad():
                student_at_start = predict_flow(
                    dit=student,
                    current=pair.start,
                    history=past_conditions,
                    condition=student_condition,
                    timestep=start_timestep,
                )
                detached_endpoint = (pair.start - student_at_start).detach()
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

            # The student sees one conditional model call. Teacher and fake are
            # evaluated at the same re-noised endpoint and contribute only their
            # DMD direction, never gradients through their parameters.
            set_requires_grad(unwrap(student), True)
            set_requires_grad(unwrap(fake), False)
            student_optimizer.zero_grad(set_to_none=True)
            student_prediction = predict_flow(
                dit=student,
                current=pair.start,
                history=past_conditions,
                condition=student_condition,
                timestep=start_timestep,
            )
            student_endpoint = pair.start - student_prediction
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
            loss = args.dmd_weight * dmd_loss + args.cauchy_weight * cauchy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap(student).parameters(), args.clip_grad_norm)
            student_optimizer.step()
            set_requires_grad(unwrap(fake), True)

            endpoint_mse = torch.nn.functional.mse_loss(student_endpoint.float(), clean_endpoint.float())
            teacher_pf_error = torch.nn.functional.mse_loss(
                teacher_flow.float(), (probe_start - probe_end).float()
            )
            metrics = {
                "step": step,
                "unit": unit,
                "stage": stage,
                "tau": tau,
                "loss": scalar_mean(loss, ctx),
                "dmd_surrogate": scalar_mean(dmd_loss, ctx),
                "cauchy": scalar_mean(cauchy_loss, ctx),
                "endpoint_mse": scalar_mean(endpoint_mse, ctx),
                "direction_rms": scalar_mean(direction_rms, ctx),
                "fake_mse": scalar_mean(torch.stack(fake_losses).mean(), ctx),
                "teacher_pf_error": scalar_mean(teacher_pf_error, ctx),
                "mean_sample_weight": scalar_mean(weights.mean(), ctx),
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
                    archive=step % args.archive_every == 0 or step == args.steps,
                    ctx=ctx,
                )
        rank0_print(ctx, f"Completed NeoDragon Pyramidal-DMD reproduction at step={args.steps}.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
