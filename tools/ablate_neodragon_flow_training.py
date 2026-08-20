#!/usr/bin/env python
"""Run a short, controlled NeoDragon flow-training ablation on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramid_flow import (
    build_pyramid_flow_state,
    corrupt_history,
    pyramid_latents,
    stage_from_ratio_slot,
)
from tools.train_neodragon_dit_bridge import dtype_from_name, load_latent_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-name", required=True)
    parser.add_argument("--flow-contract", choices=["legacy", "pyramid"], required=True)
    parser.add_argument("--history-corrupt-max", type=float, default=0.0)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-cos-weight", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--train-samples", type=int, default=80)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def require_slurm_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This ablation requires a CUDA allocation.")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Run this ablation through srun or sbatch.")


def load_records(manifest: Path) -> list[dict[str, str]]:
    rows = pd.read_csv(manifest).to_dict("records")
    return [{key: str(value) for key, value in row.items()} for row in rows]


@torch.no_grad()
def cache_conditions(
    prompts: list[str],
    *,
    text_bundle: torch.nn.Module,
    context_adapter: torch.nn.Module,
    prompt_modifier: str,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs = []
    for prompt in tqdm(prompts, desc="Cache native text conditions"):
        tokens, mask, pooled = text_bundle(prompt + prompt_modifier, device)
        outputs.append((context_adapter(tokens).cpu(), mask.cpu(), pooled.cpu()))
    return outputs


def to_device(
    condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(value.to(device=device, non_blocking=True) for value in condition)  # type: ignore[return-value]


def load_dit(
    *,
    local_model_path: str,
    dit_id: str,
    device: torch.device,
    dtype: torch.dtype,
    trainable: bool,
) -> torch.nn.Module:
    from neodragon.pyramid_mmdit import PyramidMMDiT

    model = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}",
        torch_dtype=dtype,
    ).to(device)
    model.train(trainable)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.float()
    return model


def local_timestep(
    scheduler: Any,
    *,
    stage: int,
    local_sigma: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = ((1.0 - local_sigma.float()) * (scheduler.config.num_train_timesteps - 1)).round().long()
    indices = indices.clamp(0, scheduler.config.num_train_timesteps - 1)
    return scheduler.timesteps_per_stage[stage].to(local_sigma.device)[indices].to(dtype=dtype)


def make_training_state(
    clean_high: torch.Tensor,
    *,
    flow_contract: str,
    stage: int,
    sigma: torch.Tensor,
    scheduler: Any,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flow_contract == "pyramid":
        noise_high = torch.randn(
            clean_high.shape,
            device=clean_high.device,
            dtype=clean_high.dtype,
            generator=generator,
        )
        state = build_pyramid_flow_state(
            clean_high,
            stage=stage,
            local_sigma=sigma,
            start_sigma=scheduler.start_sigmas[stage],
            end_sigma=scheduler.end_sigmas[stage],
            noise_high=noise_high,
            stages=scheduler.config.stages,
        )
        return state.noisy, state.target

    clean_stage = pyramid_latents(clean_high, stages=scheduler.config.stages)[stage]
    noise = torch.randn(
        clean_stage.shape,
        device=clean_stage.device,
        dtype=clean_stage.dtype,
        generator=generator,
    )
    amount = sigma.reshape(sigma.shape[0], 1, 1, 1, 1).to(clean_stage.dtype)
    return amount * noise + (1.0 - amount) * clean_stage, noise - clean_stage


def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.float().flatten(1)
    truth = target.float().flatten(1)
    mse = (pred - truth).square().mean(dim=1)
    power = truth.square().mean(dim=1).clamp_min(1e-8)
    return {
        "relative_mse": float((mse / power).mean().cpu()),
        "cosine": float(F.cosine_similarity(pred, truth, dim=1).mean().cpu()),
        "norm_ratio": float((pred.norm(dim=1) / truth.norm(dim=1).clamp_min(1e-8)).mean().cpu()),
    }


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


@torch.no_grad()
def evaluate_student(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    *,
    records: list[dict[str, str]],
    conditions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    condition_offset: int,
    manifest_root: Path,
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    from neodragon.utils.generation_utils import _prepare_past_condition_latents

    student.eval()
    teacher.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample_index, record in enumerate(records):
        latent = load_latent_tensor(record["latent_path"], latent_root=manifest_root)
        latent = latent.unsqueeze(0).to(device=device, dtype=dtype)
        unit_index = 1 + (sample_index * 2) % max(int(latent.shape[2]) - 1, 1)
        past = [latent[:, :, index : index + 1] for index in range(unit_index)]
        clean_high = latent[:, :, unit_index : unit_index + 1]
        tokens, mask, pooled = to_device(conditions[condition_offset + sample_index], device)
        for stage in range(scheduler.config.stages):
            histories_clean = _prepare_past_condition_latents(
                past,
                num_stages=scheduler.config.stages,
                do_classifier_free_guidance=False,
            )[stage]
            for sigma_value in (0.25, 0.75):
                sigma = torch.full((1,), sigma_value, device=device, dtype=dtype)
                noise_high = torch.randn(
                    clean_high.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                state = build_pyramid_flow_state(
                    clean_high,
                    stage=stage,
                    local_sigma=sigma,
                    start_sigma=scheduler.start_sigmas[stage],
                    end_sigma=scheduler.end_sigmas[stage],
                    noise_high=noise_high,
                    stages=scheduler.config.stages,
                )
                timestep = local_timestep(scheduler, stage=stage, local_sigma=sigma, dtype=dtype)
                for history_name, histories in (
                    ("clean", histories_clean),
                    (
                        "corrupt_1over3",
                        corrupt_history(
                            histories_clean,
                            sigma=torch.full((1,), 1.0 / 3.0, device=device, dtype=dtype),
                            generator=generator,
                        ),
                    ),
                ):
                    kwargs = {
                        "sample": [histories + [state.noisy]],
                        "encoder_hidden_states": tokens,
                        "encoder_attention_mask": mask,
                        "pooled_projections": pooled,
                        "timestep_ratio": timestep,
                    }
                    with torch.autocast("cuda", dtype=dtype):
                        student_prediction = student(**kwargs)[0]
                        teacher_prediction = teacher(**kwargs)[0]
                    student_data = prediction_metrics(student_prediction, state.target)
                    teacher_data = prediction_metrics(teacher_prediction, state.target)
                    student_teacher = prediction_metrics(student_prediction, teacher_prediction)
                    metrics = {
                        **{f"student_data_{key}": value for key, value in student_data.items()},
                        **{f"teacher_data_{key}": value for key, value in teacher_data.items()},
                        **{f"student_teacher_{key}": value for key, value in student_teacher.items()},
                    }
                    groups[f"{history_name}/stage{stage}"].append(metrics)
                    groups["overall"].append(metrics)
    return {name: average(values) for name, values in groups.items()}


@torch.no_grad()
def parameter_drift(student: torch.nn.Module, teacher: torch.nn.Module) -> dict[str, float]:
    delta_sq = 0.0
    reference_sq = 0.0
    student_sq = 0.0
    dot = 0.0
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters(), strict=True):
        student_value = student_parameter.detach().float()
        teacher_value = teacher_parameter.detach().float()
        delta_sq += float((student_value - teacher_value).square().sum().cpu())
        reference_sq += float(teacher_value.square().sum().cpu())
        student_sq += float(student_value.square().sum().cpu())
        dot += float((student_value * teacher_value).sum().cpu())
    return {
        "relative_l2": float((delta_sq / max(reference_sq, 1e-12)) ** 0.5),
        "cosine": float(dot / max((reference_sq * student_sq) ** 0.5, 1e-12)),
    }


def main() -> None:
    args = parse_args()
    require_slurm_cuda()
    if args.data_weight <= 0.0 and args.teacher_weight <= 0.0 and args.teacher_cos_weight <= 0.0:
        raise ValueError("At least one training loss must be enabled.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir).expanduser().resolve() / args.cell_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).expanduser().resolve()
    all_records = load_records(manifest)
    if len(all_records) < args.train_samples + args.eval_samples:
        raise ValueError(
            f"Need {args.train_samples + args.eval_samples} latents, found {len(all_records)}"
        )
    train_records = all_records[: args.train_samples]
    eval_records = all_records[-args.eval_samples :]
    condition_records = train_records + eval_records

    cfg = load_config(args.config)
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER, _prepare_past_condition_latents

    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    conditions = cache_conditions(
        [record["prompt"] for record in condition_records] + [""],
        text_bundle=text_bundle,
        context_adapter=context_adapter,
        prompt_modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
    )
    null_condition = conditions[-1]
    del text_bundle, context_adapter
    torch.cuda.empty_cache()

    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    student = load_dit(
        local_model_path=local_model_path,
        dit_id=MULTISTEP_DIT_ID,
        device=device,
        dtype=dtype,
        trainable=True,
    )
    teacher = None
    if args.teacher_weight > 0.0 or args.teacher_cos_weight > 0.0:
        teacher = load_dit(
            local_model_path=local_model_path,
            dit_id=MULTISTEP_DIT_ID,
            device=device,
            dtype=dtype,
            trainable=False,
        )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 100_000)
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), desc=f"NeoDragon ablation {args.cell_name}")
    for step in progress:
        record_index = (step - 1) % len(train_records)
        record = train_records[record_index]
        latent = load_latent_tensor(record["latent_path"], latent_root=manifest.parent)
        latent = latent.unsqueeze(0).to(device=device, dtype=dtype)
        unit_index = int(
            torch.randint(1, latent.shape[2], (1,), device=device, generator=generator).item()
        )
        # Keep this historical pilot's middle-stage-heavy mixture reproducible.
        stage = stage_from_ratio_slot(step - 1, (1, 2, 1))
        sigma = torch.rand((1,), device=device, dtype=dtype, generator=generator)
        timestep = local_timestep(scheduler, stage=stage, local_sigma=sigma, dtype=dtype)
        clean_high = latent[:, :, unit_index : unit_index + 1]
        noisy, target = make_training_state(
            clean_high,
            flow_contract=args.flow_contract,
            stage=stage,
            sigma=sigma,
            scheduler=scheduler,
            generator=generator,
        )
        past_units = [latent[:, :, index : index + 1] for index in range(unit_index)]
        histories = _prepare_past_condition_latents(
            past_units,
            num_stages=scheduler.config.stages,
            do_classifier_free_guidance=False,
        )[stage]
        if args.history_corrupt_max > 0.0:
            history_sigma = torch.rand((1,), device=device, dtype=dtype, generator=generator)
            history_sigma = history_sigma * args.history_corrupt_max
            histories = corrupt_history(histories, sigma=history_sigma, generator=generator)

        use_null = bool(
            torch.rand((1,), device=device, generator=generator).item() < args.condition_dropout
        )
        condition = null_condition if use_null else conditions[record_index]
        tokens, mask, pooled = to_device(condition, device)
        kwargs = {
            "sample": [histories + [noisy]],
            "encoder_hidden_states": tokens,
            "encoder_attention_mask": mask,
            "pooled_projections": pooled,
            "timestep_ratio": timestep,
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            prediction = student(**kwargs)[0]
            data_loss = F.mse_loss(prediction.float(), target.float())
            teacher_loss = prediction.new_zeros((), dtype=torch.float32)
            teacher_cos_loss = prediction.new_zeros((), dtype=torch.float32)
            if teacher is not None:
                with torch.no_grad():
                    teacher_prediction = teacher(**kwargs)[0]
                teacher_loss = F.mse_loss(prediction.float(), teacher_prediction.float())
                teacher_cos_loss = (
                    1.0
                    - F.cosine_similarity(
                        prediction.float().flatten(1),
                        teacher_prediction.float().flatten(1),
                        dim=1,
                    ).mean()
                )
            loss = (
                args.data_weight * data_loss
                + args.teacher_weight * teacher_loss
                + args.teacher_cos_weight * teacher_cos_loss
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
        optimizer.step()
        row = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "data": float(data_loss.detach().cpu()),
            "teacher": float(teacher_loss.detach().cpu()),
            "teacher_cos": float(teacher_cos_loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "stage": float(stage),
            "unit": float(unit_index),
            "sigma": float(sigma.float().mean().cpu()),
        }
        history.append(row)
        if step % args.log_every == 0 or step == args.steps:
            recent = history[-args.log_every :]
            display = average([{key: value for key, value in item.items() if key not in {"step", "stage", "unit"}} for item in recent])
            progress.set_postfix({key: f"{value:.4f}" for key, value in display.items() if key in {"loss", "data", "teacher", "grad_norm"}})

    if teacher is None:
        teacher = load_dit(
            local_model_path=local_model_path,
            dit_id=MULTISTEP_DIT_ID,
            device=device,
            dtype=dtype,
            trainable=False,
        )
    eval_metrics = evaluate_student(
        student,
        teacher,
        records=eval_records,
        conditions=conditions,
        condition_offset=len(train_records),
        manifest_root=manifest.parent,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        seed=args.seed + 200_000,
    )
    drift = parameter_drift(student, teacher)
    elapsed = time.perf_counter() - start_time
    report = {
        "cell": args.cell_name,
        "flow_contract": args.flow_contract,
        "history_corrupt_max": args.history_corrupt_max,
        "data_weight": args.data_weight,
        "teacher_weight": args.teacher_weight,
        "teacher_cos_weight": args.teacher_cos_weight,
        "steps": args.steps,
        "lr": args.lr,
        "stage_sampling": "1:2:1 deterministic slots",
        "condition_dropout": args.condition_dropout,
        "elapsed_seconds": elapsed,
        "last_50": average(history[-min(50, len(history)) :]),
        "parameter_drift": drift,
        "evaluation": eval_metrics,
    }
    state = {key: value.detach().to(device="cpu", dtype=torch.bfloat16) for key, value in student.state_dict().items()}
    checkpoint = {
        "dit": state,
        "teacher_stack": {"name": "multistep"},
        "ablation": report,
    }
    torch.save(checkpoint, output_dir / "neodragon_ablation_dit.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
