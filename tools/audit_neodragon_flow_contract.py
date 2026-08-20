#!/usr/bin/env python
"""Audit whether NeoDragon is evaluated on its native pyramidal-flow contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.neodragon_pyramid_flow import (
    build_pyramid_flow_state,
    corrupt_history,
    pyramid_latents,
)
from tools.train_neodragon_dit_bridge import dtype_from_name, load_latent_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--sigma", action="append", type=float, default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def require_slurm_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires a CUDA allocation.")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Run this audit through srun or sbatch.")


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path | None]]:
    outputs: list[tuple[str, Path | None]] = [("released_multistep", None)]
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip())
        path = Path(raw_path).expanduser().resolve()
        if not label or not path.is_file():
            raise FileNotFoundError(path)
        outputs.append((label, path))
    return outputs


def load_records(manifest: Path, max_samples: int) -> list[dict[str, str]]:
    import pandas as pd

    rows = pd.read_csv(manifest).head(max_samples).to_dict("records")
    if len(rows) < 2:
        raise ValueError("Flow audit needs at least two records for shuffled-condition tests.")
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
    for prompt in prompts:
        tokens, mask, pooled = text_bundle(prompt + prompt_modifier, device)
        tokens = context_adapter(tokens)
        outputs.append((tokens.cpu(), mask.cpu(), pooled.cpu()))
    return outputs


def load_model(
    *,
    local_model_path: str,
    dit_id: str,
    checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from neodragon.pyramid_mmdit import PyramidMMDiT

    model = PyramidMMDiT.from_pretrained(
        f"{local_model_path}/{dit_id}",
        torch_dtype=dtype,
    )
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        state = payload.get("dit", payload.get("student_state", payload))
        if not isinstance(state, dict):
            raise ValueError(f"Could not find DiT state in {checkpoint}")
        model.load_state_dict(state, strict=True)
        del payload, state
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.float().flatten(1)
    truth = target.float().flatten(1)
    mse = (pred - truth).square().mean(dim=1)
    target_power = truth.square().mean(dim=1).clamp_min(1e-8)
    return {
        "mse": float(mse.mean().cpu()),
        "relative_mse": float((mse / target_power).mean().cpu()),
        "cosine": float(F.cosine_similarity(pred, truth, dim=1).mean().cpu()),
        "norm_ratio": float(
            (pred.norm(dim=1) / truth.norm(dim=1).clamp_min(1e-8)).mean().cpu()
        ),
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
    }


@torch.no_grad()
def audit_model(
    model: torch.nn.Module,
    *,
    records: list[dict[str, str]],
    conditions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    manifest_root: Path,
    scheduler: Any,
    sigmas: list[float],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from neodragon.utils.generation_utils import _prepare_past_condition_latents

    rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample_index, record in enumerate(records):
        latent = load_latent_tensor(record["latent_path"], latent_root=manifest_root)
        latent = latent.unsqueeze(0).to(device=device, dtype=dtype)
        unit_index = 1 + (sample_index * 2) % max(int(latent.shape[2]) - 1, 1)
        clean_high = latent[:, :, unit_index : unit_index + 1]
        past_units = [latent[:, :, index : index + 1] for index in range(unit_index)]
        prompt_condition = tuple(value.to(device) for value in conditions[sample_index])
        shuffled_condition = tuple(
            value.to(device) for value in conditions[(sample_index + 1) % len(conditions)]
        )

        for stage in range(scheduler.config.stages):
            clean_stage = pyramid_latents(clean_high, stages=scheduler.config.stages)[stage]
            for local_sigma in sigmas:
                sigma_tensor = torch.full((1,), local_sigma, device=device, dtype=dtype)
                noise_high = torch.randn(
                    clean_high.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                proper = build_pyramid_flow_state(
                    clean_high,
                    stage=stage,
                    local_sigma=sigma_tensor,
                    start_sigma=scheduler.start_sigmas[stage],
                    end_sigma=scheduler.end_sigmas[stage],
                    noise_high=noise_high,
                    stages=scheduler.config.stages,
                )
                legacy_noise = torch.randn(
                    clean_stage.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                legacy_noisy = sigma_tensor.reshape(1, 1, 1, 1, 1) * legacy_noise + (
                    1.0 - sigma_tensor.reshape(1, 1, 1, 1, 1)
                ) * clean_stage
                legacy_target = legacy_noise - clean_stage

                timestep_index = min(
                    max(int(round((1.0 - local_sigma) * (scheduler.config.num_train_timesteps - 1))), 0),
                    scheduler.config.num_train_timesteps - 1,
                )
                timestep = scheduler.timesteps_per_stage[stage][timestep_index].to(
                    device=device,
                    dtype=dtype,
                ).expand(1)

                for history_mode in ("clean", "corrupt_1over3"):
                    histories = _prepare_past_condition_latents(
                        past_units,
                        num_stages=scheduler.config.stages,
                        do_classifier_free_guidance=False,
                    )[stage]
                    if history_mode != "clean":
                        histories = corrupt_history(
                            histories,
                            sigma=torch.full((1,), 1.0 / 3.0, device=device, dtype=dtype),
                            generator=generator,
                        )

                    for path_name, noisy, target in (
                        ("proper_pyramid", proper.noisy, proper.target),
                        ("legacy_direct", legacy_noisy, legacy_target),
                    ):
                        tokens, mask, pooled = prompt_condition
                        prediction = model(
                            sample=[histories + [noisy]],
                            encoder_hidden_states=tokens,
                            encoder_attention_mask=mask,
                            pooled_projections=pooled,
                            timestep_ratio=timestep,
                        )[0]
                        shuffled_tokens, shuffled_mask, shuffled_pooled = shuffled_condition
                        shuffled_prediction = model(
                            sample=[histories + [noisy]],
                            encoder_hidden_states=shuffled_tokens,
                            encoder_attention_mask=shuffled_mask,
                            pooled_projections=shuffled_pooled,
                            timestep_ratio=timestep,
                        )[0]
                        metrics = tensor_metrics(prediction, target)
                        condition_delta = (
                            (prediction.float() - shuffled_prediction.float()).square().mean().sqrt()
                            / prediction.float().square().mean().sqrt().clamp_min(1e-8)
                        )
                        metrics["condition_delta_ratio"] = float(condition_delta.cpu())
                        target_cosine = F.cosine_similarity(
                            proper.target.float().flatten(1),
                            legacy_target.float().flatten(1),
                            dim=1,
                        ).mean()
                        metrics["proper_legacy_target_cosine"] = float(target_cosine.cpu())
                        row = {
                            "sample_index": sample_index,
                            "unit_index": unit_index,
                            "stage": stage,
                            "local_sigma": local_sigma,
                            "history": history_mode,
                            "path": path_name,
                            **metrics,
                        }
                        rows.append(row)
                        grouped[f"{path_name}/{history_mode}/stage{stage}"].append(metrics)

    summary = {name: average_metrics(values) for name, values in grouped.items()}
    summary["overall"] = {}
    for path_name in ("proper_pyramid", "legacy_direct"):
        subset = [
            {key: float(row[key]) for key in (
                "mse",
                "relative_mse",
                "cosine",
                "norm_ratio",
                "condition_delta_ratio",
                "proper_legacy_target_cosine",
            )}
            for row in rows
            if row["path"] == path_name
        ]
        summary["overall"][path_name] = average_metrics(subset)
    return rows, summary


def main() -> None:
    args = parse_args()
    require_slurm_cuda()
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).expanduser().resolve()
    records = load_records(manifest, args.max_samples)
    checkpoints = parse_checkpoints(args.checkpoint)
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
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{MULTISTEP_CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    conditions = cache_conditions(
        [record["prompt"] for record in records],
        text_bundle=text_bundle,
        context_adapter=context_adapter,
        prompt_modifier=DEFAULT_PROMPT_MODIFIER,
        device=device,
    )
    del text_bundle, context_adapter
    torch.cuda.empty_cache()

    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    sigmas = args.sigma or [0.2, 0.5, 0.8]
    report: dict[str, Any] = {
        "objective": "neodragon_flow_contract_audit",
        "manifest": str(manifest),
        "samples": len(records),
        "sigmas": sigmas,
        "stage_ratio_reference": [1, 2, 1],
        "models": {},
    }
    for model_index, (label, checkpoint) in enumerate(checkpoints):
        print(f"Auditing {label} ({checkpoint or 'released multistep'})", flush=True)
        model = load_model(
            local_model_path=local_model_path,
            dit_id=MULTISTEP_DIT_ID,
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        rows, summary = audit_model(
            model,
            records=records,
            conditions=conditions,
            manifest_root=manifest.parent,
            scheduler=scheduler,
            sigmas=sigmas,
            device=device,
            dtype=dtype,
            seed=args.seed + model_index * 10_000,
        )
        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report["models"][label] = summary
        del model
        torch.cuda.empty_cache()

    (output_dir / "flow_contract_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
