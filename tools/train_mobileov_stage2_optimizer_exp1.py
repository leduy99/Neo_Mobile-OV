#!/usr/bin/env python
"""Advisor Exp 1: original epoch-3 Adam -> 20K batch-64 updates, LR decay + EMA."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tools import train_mobileov_stage1 as legacy
from tools.data_prepare.download_alignment_images import atomic_json, output_lock, sha256_file
from new_mobile_ov.training import stage2_optimizer_exp1 as exp1
from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training import stage2_epoch_extension as extension
from new_mobile_ov.training.distributed import setup_distributed, cleanup_distributed, barrier, scalar_mean

PARENT = Path("output/mobileov_stage2_media3epochs/17641364/stage2_resume.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-resume", type=Path, default=PARENT)
    parser.add_argument("--resume", type=Path, help="Own Exp-1 full-state checkpoint; never an EMA inference file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("download_data/data/univideo_stage1_expanded650"))
    parser.add_argument("--config", type=Path, default=Path("configs/mobile_ov_neodragon.yaml"))
    parser.add_argument("--steps", type=int, default=20000, help="Additional optimizer updates, not media steps")
    parser.add_argument("--decay-steps", type=int, default=20000)
    parser.add_argument("--ema-decay", type=float, default=.9995)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--expected-parent-step", type=int, default=283977)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=2000, help="Save the full Adam/EMA/RNG state")
    parser.add_argument("--evaluate-every", type=int, default=10000, help="Archive RAW + EMA inference weights")
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, help="Pause after this many additional Exp-1 updates")
    parser.add_argument("--max-runtime-seconds", type=float, default=244800)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--integration-smoke", action="store_true", help="Tiny CPU-test budget only; no quality claim")
    args = parser.parse_args(argv)
    if (min(args.steps, args.decay_steps, args.save_every, args.evaluate_every, args.validate_every,
            args.validation_samples, args.log_every, args.expected_parent_step) < 1
            or args.decay_steps > args.steps or args.workers < 0 or args.max_runtime_seconds < 0):
        parser.error("Invalid budget, interval, worker count or runtime")
    if args.accumulation not in (4, 8) or not 0 < args.ema_decay < 1:
        parser.error("Exp 1 requires accumulation 4 (16 GPUs) or 8 (8 GPUs) and EMA decay in (0,1)")
    if args.integration_smoke:
        if args.steps > 10:
            parser.error("Integration smoke is limited to ten additional updates")
    elif (args.steps, args.decay_steps, args.ema_decay, args.expected_parent_step) != (20000, 20000, .9995, 283977):
        parser.error("Production Exp 1 is pinned to epoch 3, 20K updates/decay and EMA .9995")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("Stop-after is an additional-update count within the Exp-1 budget")
    return args


def file_digest(path, ctx):
    value = [sha256_file(path) if ctx.is_main else None]
    if ctx.is_distributed:
        dist.broadcast_object_list(value, src=0)
    return value[0]


def latest_link(source, destination):
    temporary = destination.with_suffix(".pt.link")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def save(args, ctx, modules, optimizer, ema, contract, update, *, archive):
    rng = legacy.capture_rng_states(ctx)
    if not ctx.is_main:
        return
    origin = contract["optimizer_exp1"]["parent_step"]
    state = {name: exp1.cpu_weights(module) for name, module in modules.items()}
    payload = dict(format=stage2.FORMAT, step=origin + update, exp1_update=update,
                   connector_spec=modules["connector"].spec, contract=contract, **state)
    if archive:
        manifest_path = args.output_dir / "evaluation_checkpoints.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
            format=exp1.POLICY, default_weights="ema", checkpoints={})
        for variant in ("raw", "ema"):
            path = args.output_dir / f"stage2_dit_connector_{variant}_update{update:06d}.pt"
            weights = state if variant == "raw" else ema.shadow
            legacy.atomic_torch_save(dict(payload, **weights, weight_variant=variant, ema_updates=ema.updates), path)
            latest = args.output_dir / ("stage2_dit_connector_latest.pt" if variant == "ema"
                                       else "stage2_dit_connector_raw_latest.pt")
            latest_link(path, latest)
            manifest["checkpoints"][f"{variant}_{update}"] = dict(
                path=path.name, sha256=sha256_file(path), step=origin + update,
                exp1_update=update, weight_variant=variant, ema_updates=ema.updates)
        atomic_json(manifest_path, manifest)
    resume = dict(payload, format=exp1.FORMAT, optimizer=optimizer.state_dict(), ema=ema.state_dict(),
                  scheduler=dict(contract["lr_schedule"], completed_updates=update),
                  sampler_cursor=exp1.media_cursor(contract, update), rng_states=rng)
    legacy.atomic_torch_save(resume, args.output_dir / "exp1_resume.pt")
    print(f"Saved Exp1 update={update}, Adam step={origin + update}, EMA updates={ema.updates}; "
          f"raw+EMA archive={archive}.", flush=True)


def validation(args, ctx, encoder, modules, vae, scheduler, dataset, ema, *, include_ema):
    rng = legacy.capture_rng_states(ctx)
    try:
        raw = legacy.validate(args, ctx, encoder, modules["connector"], modules["dit"], vae, scheduler, dataset)
        metrics = dict(raw=raw)
        if include_ema:
            legacy.restore_rng_states(rng, ctx)
            with exp1.ema_weights(modules, ema, ctx):
                metrics["ema"] = legacy.validate(args, ctx, encoder, modules["connector"], modules["dit"],
                                                 vae, scheduler, dataset)
        return metrics
    finally:
        legacy.restore_rng_states(rng, ctx)


def train(args, ctx):
    launched = time.monotonic()
    if not args.integration_smoke and (ctx.world_size, args.accumulation) not in ((16, 4), (8, 8)):
        raise ValueError("Production Exp 1 requires 16 GPUs x4 or 8 GPUs x8; global batch 64")
    source = args.resume or args.parent_resume
    if args.resume is None and args.output_dir.resolve() == source.resolve().parent:
        raise ValueError("Use a separate Exp-1 output; never overwrite the epoch-3 run")
    existing = args.output_dir / "run_contract.json"
    if existing.exists() and (args.resume is None or args.resume.resolve() != (args.output_dir / "exp1_resume.pt").resolve()):
        raise ValueError("Existing Exp-1 output must resume its own exp1_resume.pt; never rewind it")
    if args.resume is not None and not existing.is_file():
        raise ValueError("Own-run resume requires the original Exp-1 run_contract.json")
    if ctx.is_main:
        legacy.verify_release(args.data_root)
    barrier()
    saved = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if args.resume:
        parent = saved.get("contract", {}).get("parent_contract", {})
        parent_sha = saved.get("contract", {}).get("parent_resume_sha256")
        if not isinstance(parent_sha, str) or len(parent_sha) != 64:
            raise ValueError("Missing epoch-3 provenance in Exp-1 resume")
        start = saved.get("exp1_update", -1)
    else:
        exp1.validate_parent(saved, expected_step=args.expected_parent_step)
        parent, parent_sha, start = saved["contract"], file_digest(source, ctx), 0
    if parent.get("steps") != args.expected_parent_step:
        raise ValueError("Parent step differs from the pinned epoch-3 source")
    if sha256_file(args.data_root / "stage1_summary.json") != parent.get("data_summary_sha256"):
        raise ValueError("Exp 1 must use the unchanged expanded650 release")
    if sha256_file(args.config) != parent.get("config_sha256"):
        raise ValueError("Exp 1 must use the parent's exact model config")
    for key in ("processor", "max_tokens", "condition_dropout", "short_side", "long_side", "seed", "vae_window_size"):
        setattr(args, key, parent[key])
    args.phase, args.media_epochs = 2, 3  # Preserve the legacy validation's five task routes.
    args.reconstruction_weight = .1
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg, encoder, connector, dit, vae, scheduler = legacy.load_stack(args, ctx.device)
    signatures = legacy.frozen_signatures(dict(smolvlm2=encoder, dit=dit, vae=vae))
    if signatures != parent.get("initial_weights_sha256") or encoder.processor_sha256 != parent.get("processor_sha256"):
        raise ValueError("Native initialization, frozen SmolVLM2/VAE or processor differs from epoch 3")
    contract = exp1.continuation_contract(parent, parent_sha256=parent_sha, world_size=ctx.world_size,
        accumulation=args.accumulation, total=args.steps, decay_steps=args.decay_steps, ema_decay=args.ema_decay)
    contract["exp1_implementation_sha256"] = {name: sha256_file(ROOT / name) for name in (
        "tools/train_mobileov_stage2_optimizer_exp1.py", "new_mobile_ov/training/stage2_optimizer_exp1.py",
        "tools/train_mobileov_stage1.py", "new_mobile_ov/training/stage2_alignment.py",
        "new_mobile_ov/training/stage1_alignment.py", "new_mobile_ov/training/stage1_alignment_data.py",
        "new_mobile_ov/training/stage2_media_epochs.py", "new_mobile_ov/training/neodragon_pyramid_flow.py",
        "new_mobile_ov/bridge/stage1_conditioner.py", "new_mobile_ov/training/stage1_vae.py",
        "tools/train_neodragon_dit_bridge.py", "new_mobile_ov/training/distributed.py")}
    if existing.exists() and json.loads(existing.read_text()) != contract:
        raise ValueError("Own-run output contract differs")
    load_args = argparse.Namespace(**dict(vars(args), expected_stage2_step=saved["step"]))
    stage2.initialize_from_stage2(dict(saved, format=stage2.FORMAT), connector, dit, contract=contract, args=load_args)
    modules = dict(connector=connector, dit=dit)
    trainable = stage2.JointFlowModel(connector, dit)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=2e-5, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0, foreach=False)
    audit = (exp1.validate_own_resume(saved, contract, parameters=trainable.parameters()) if args.resume else
             exp1.validate_parent(saved, expected_step=args.expected_parent_step, parameters=trainable.parameters()))
    optimizer.load_state_dict(saved["optimizer"])
    ema = exp1.FP32EMA(modules, args.ema_decay) if ctx.is_main else None
    if args.resume and ctx.is_main:
        ema.load_state_dict(saved["ema"], expected_updates=start)
    resume_rng = saved.get("rng_states") if args.resume else None
    del saved
    end = args.stop_after if args.stop_after is not None else args.steps
    if not 0 <= start < end <= args.steps:
        raise ValueError("Invalid, already completed or rewound Exp-1 update range")
    model = DDP(trainable, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True) if ctx.is_distributed else trainable
    dataset = exp1.GroupedMediaDataset(args.data_root, parent=parent, rank=ctx.rank, world_size=ctx.world_size,
        accumulation=args.accumulation, total=args.steps, start_update=start,
        short_side=args.short_side, long_side=args.long_side)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None, timeout=120 if args.workers else 0,
        prefetch_factor=2 if args.workers else None)
    batches = iter(loader)
    if ctx.is_main:
        atomic_json(existing, contract)
        atomic_json(args.output_dir / f"resume_audit_update{start:06d}.json", dict(audit,
            exp1_update=start, optimizer_source=str(source), parent_resume_sha256=parent_sha,
            source_batch=parent["media_epochs"]["global_batch"], global_batch=ctx.world_size * args.accumulation,
            exact_parent_trajectory=False, raw_weights_restored=True, ema_reinitialized=not bool(args.resume)))
        (args.output_dir / "training_paused.json").unlink(missing_ok=True)
        (args.output_dir / "training_complete.json").unlink(missing_ok=True)
        print(f"Exp1 updates {start}->{args.steps}; parent Adam step {args.expected_parent_step}; "
              f"batch={ctx.world_size * args.accumulation}; unchanged data/loss, no warmup/history noise/LoRA. "
              "CPU FP32 EMA on rank0; inference latest defaults to EMA.", flush=True)
    metrics = validation(args, ctx, encoder, modules, vae, scheduler, dataset, ema, include_ema=False)
    if ctx.is_main:
        atomic_json(args.output_dir / f"validation_update{start:06d}.json", metrics)
    if resume_rng is not None:
        legacy.restore_rng_states(resume_rng, ctx)
    for update in range(start + 1, end + 1):
        trainable.train()
        optimizer.zero_grad(set_to_none=True)
        lr = exp1.learning_rate(update, total=args.steps, decay_steps=args.decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, observations = {task: [] for task in media.METRIC_TASKS}, []
        weighted = torch.zeros((), device=ctx.device)
        for micro in range(args.accumulation):
            sample = next(batches)
            rng = legacy.generator_for(ctx.device, args.seed, sample["micro"], sample["rng_rank"])
            layers, mask, latent, dropped = legacy.encode_sample(encoder, vae, sample, ctx.device, rng,
                args.condition_dropout, window_size=args.vae_window_size)
            stage = int(torch.randint(3, (), device=ctx.device, generator=rng))
            mode = sample.get("mode", sample["task"])
            unit = media.sample_unit(mode, latent.shape[2], ctx.device, rng)
            scale = sample["loss_scale"] / args.accumulation
            sync = model.no_sync() if ctx.is_distributed and micro + 1 < args.accumulation else nullcontext()
            with sync, legacy.autocast(ctx.device):
                loss = model(layers, mask, latent, scheduler, stage=stage, unit=unit, generator=rng)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Non-finite flow loss: {sample['sample_id']}")
                (loss * scale).backward()
                weighted += loss.detach() * scale
                if not sample["padding"]:
                    losses[mode].append(loss.detach())
                if sample["task"] == "t2i":
                    rec_rng = legacy.generator_for(ctx.device, args.seed + stage2.RECONSTRUCTION_SEED_OFFSET,
                                                    sample["micro"], sample["rng_rank"])
                    rec_layers, rec_mask = stage2.reconstruction_condition(encoder, sample)
                    rec_stage = int(torch.randint(3, (), device=ctx.device, generator=rec_rng))
                    rec = model(rec_layers, rec_mask, latent, scheduler, stage=rec_stage, unit=0, generator=rec_rng)
                    if not bool(torch.isfinite(rec)):
                        raise RuntimeError("Non-finite paired reconstruction loss")
                    (rec * (2 * args.reconstruction_weight * scale)).backward()
                    weighted += rec.detach() * (2 * args.reconstruction_weight * scale)
                    if not sample["padding"]:
                        losses["image_reconstruction"].append(rec.detach())
                    del rec, rec_layers, rec_mask
            observations.append(dict(task=mode, sample_id=sample["sample_id"], unit=unit, stage=stage,
                dropped=dropped, tokens=mask.shape[1], epoch=sample["epoch"] + 1, padding=sample["padding"],
                source_duration_seconds=sample.get("source_duration_seconds"),
                effective_sample_fps=sample.get("effective_sample_fps"),
                original_micro=sample["micro"], rng_rank=sample["rng_rank"], loss_scale=sample["loss_scale"]))
            del layers, mask, latent, loss
        norms = {name: stage2.gradient_norm(module) for name, module in modules.items()}
        if any(not bool(value > 0) for value in norms.values()):
            raise RuntimeError("Connector or DiT received no gradient")
        if any(p.grad is not None for module in (encoder, vae) for p in module.parameters()):
            raise RuntimeError("Frozen SmolVLM2/VAE received gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if ctx.is_main:
            ema.update(modules)
        optimizer.zero_grad(set_to_none=True)
        if update == start + 1 and ctx.is_main:
            atomic_json(args.output_dir / f"first_update_optimizer_audit_update{update:06d}.json",
                dict(extension.optimizer_audit(optimizer.state_dict(), args.expected_parent_step + update, lr),
                     exp1_update=update, ema_updates=ema.updates))
        if legacy.runtime_expired(args, ctx, launched):
            end = update
        if update == start + 1 or update % args.log_every == 0 or update == end:
            means, counts = media.distributed_loss_means(losses, ctx)
            report = dict(step=args.expected_parent_step + update, exp1_update=update, lr=lr, ema_updates=update,
                global_batch=ctx.world_size * args.accumulation,
                sampler_cursor=exp1.media_cursor(contract, update),
                original_media_epochs=exp1.media_cursor(contract, update) / parent["media_epochs"]["steps_per_epoch"],
                loss=means, valid_samples_in_update=counts, weighted_loss=scalar_mean(weighted, ctx),
                grad_norm=scalar_mean(norm, ctx), component_grad_norm={k: scalar_mean(v, ctx) for k, v in norms.items()},
                elapsed_seconds=time.monotonic() - launched, rank0_samples=observations, memory_by_rank=legacy.memory_report(ctx))
            if ctx.is_main:
                print(json.dumps(report), flush=True)
                with (args.output_dir / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(report, allow_nan=False) + "\n")
        archive = update % args.evaluate_every == 0 or update == args.steps
        if update % args.validate_every == 0 or archive or update == end:
            metrics = validation(args, ctx, encoder, modules, vae, scheduler, dataset, ema, include_ema=archive)
            if ctx.is_main:
                atomic_json(args.output_dir / f"validation_update{update:06d}.json", metrics)
        if update % args.save_every == 0 or archive or update == end:
            barrier()
            save(args, ctx, modules, optimizer, ema, contract, update, archive=archive)
            barrier()
        if update == end:
            break
    if ctx.is_main:
        complete = end == args.steps
        atomic_json(args.output_dir / ("training_complete.json" if complete else "training_paused.json"),
            dict(format=exp1.POLICY, exp1_update=end, step=args.expected_parent_step + end,
                 ema_updates=ema.updates, quality_claim="none; compare raw and EMA generated videos"))
        print(f"{'Completed' if complete else 'Paused'} Exp1 after {end} additional updates.", flush=True)


@record
def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit training inside a SLURM GPU job, not on a login node")
    ctx = setup_distributed()
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("A GPU allocation is required")
        with output_lock(args.output_dir) if ctx.is_main else nullcontext():
            train(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
