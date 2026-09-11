#!/usr/bin/env python
"""UniVideo-inspired connector-only alignment adapted to NeoDragon's pyramid.

This is not an exact UniVideo reproduction: public data, MCP connector, pooled
head, native pyramid schedule, equal task weights, and 1/49-frame targets.
No native text teacher, DreamLite, external anchor, DiT updates, or DMD.
UniVideo section 2.4 includes T2V but Table 7 lists one frame for Stage 1;
including video here follows our explicit three-task scope, not that table.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from new_mobile_ov.bridge.stage1_conditioner import FrozenStage1Encoder, Stage1Connector
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import setup_distributed, cleanup_distributed, barrier, scalar_mean
from new_mobile_ov.training.stage1_alignment import (
    TASKS, Stage1Dataset, JsonlRecords, prepare_sample, verify_release, flow_loss, frozen_signatures,
)
from tools.data_prepare.download_alignment_images import atomic_json, output_lock, sha256_file
from tools.train_neodragon_dit_bridge import load_neodragon_train_modules, scale_vae_latents

FORMAT = "mobileov_stage1_mcp_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--data-root", type=Path, default=Path("download_data/data/univideo_stage1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--steps", type=int, default=15000, help="Total optimizer updates, not microbatches")
    parser.add_argument("--accumulation", type=int, default=3, help="Multiple of 3; equal T2I/T2V/reconstruction weight")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--archive-every", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=2, help="Per task per rank; 3 stages, first/last video units")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Pause at this optimizer step to test exact resume; not a new recipe")
    parser.add_argument("--verify-frozen", action="store_true", help="Full hashes before/after; intended for local smoke")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.accumulation, args.save_every, args.archive_every,
           args.validate_every, args.validation_samples, args.log_every, args.max_tokens) < 1:
        parser.error("Counts must be positive")
    if args.accumulation % 3 or args.workers < 0 or args.warmup < 0:
        parser.error("Accumulation must be divisible by 3; workers/warmup cannot be negative")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("Stop-after must be within the planned run")
    if not 0 <= args.condition_dropout < 1 or not 0 < args.lr < 1:
        parser.error("Invalid condition dropout or learning rate")
    if min(args.short_side, args.long_side) < 64 or args.short_side % 64 or args.long_side % 64:
        parser.error("Spatial sizes must be multiples of 64 (VAE, pyramid, patchification)")
    return args


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def load_stack(args, device):
    cfg = load_config(args.config)
    encoder = FrozenStage1Encoder(cfg.bridge.smolvlm2_ckpt_path, args.processor,
                                  device, torch.bfloat16, args.max_tokens)
    dit, vae, scheduler, _ = load_neodragon_train_modules(cfg, device, torch.bfloat16, target_stack="multistep")
    dit.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    connector = Stage1Connector(input_dim=encoder.input_dim).to(device=device, dtype=torch.float32)
    return cfg, encoder, connector, dit, vae, scheduler


def encode_sample(encoder, vae, sample, device, generator, dropout):
    drop = bool(torch.rand((), device=device, generator=generator) < dropout)
    layers, mask = encoder(sample["prompt"], sample["image"], drop_condition=drop)
    with torch.no_grad(), autocast(device):
        video = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        posterior = vae.encode(video, temporal_chunk=video.shape[2] > 1).latent_dist
        latent = scale_vae_latents(posterior.sample(generator=generator)).to(torch.bfloat16)
    expected = 1 if sample["task"] != "t2v" else 7
    if latent.shape[2] != expected or not bool(torch.isfinite(latent).all()):
        raise ValueError(f"Invalid VAE latent for {sample['task']}: {latent.shape}")
    return layers, mask, latent, drop


def generator_for(device, seed, ordinal, rank):
    return torch.Generator(device=device).manual_seed(seed + 1000003 * ordinal + 9176 * rank)


def validate(args, ctx, encoder, connector, dit, vae, scheduler, dataset):
    metrics = {}
    connector.eval()
    with torch.no_grad():
        for task in TASKS:
            records = JsonlRecords(args.data_root / "validation" / f"{task}.jsonl")
            values = {}
            for item in range(args.validation_samples):
                index = (item * ctx.world_size + ctx.rank) % len(records)
                sample = prepare_sample(records[index], dataset.sources, args.short_side, args.long_side)
                rng = generator_for(ctx.device, args.seed + 888888, index, 0)
                layers, mask, latent, _ = encode_sample(encoder, vae, sample, ctx.device, rng, 0)
                units = [0] if task != "t2v" else [0, latent.shape[2] - 1]
                for stage in range(3):
                    for unit in units:
                        with autocast(ctx.device):
                            loss = flow_loss(dit, connector, layers, mask, latent, scheduler,
                                             stage=stage, unit=unit, generator=rng, gradient_checkpointing=False)
                        key = f"{task}.stage{stage}.{'first' if unit == 0 else 'last'}"
                        values.setdefault(key, []).append(loss)
            for key, losses in values.items():
                value = scalar_mean(torch.stack(losses).mean(), ctx)
                if not torch.isfinite(torch.tensor(value)):
                    raise RuntimeError(f"Non-finite validation loss: {key}")
                metrics[key] = value
    connector.train()
    return metrics


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save(args, step, connector, optimizer, contract, *, final=False):
    payload = dict(format=FORMAT, step=step, connector_spec=connector.spec,
                   connector={key: value.detach().cpu() for key, value in connector.state_dict().items()},
                   contract=contract)
    atomic_torch_save(payload, args.output_dir / "stage1_connector_latest.pt")
    if step % args.archive_every == 0 or final:
        atomic_torch_save(payload, args.output_dir / f"stage1_connector_step{step:06d}.pt")
    atomic_torch_save({**payload, "optimizer": optimizer.state_dict()}, args.output_dir / "stage1_resume.pt")
    print(f"Saved Stage-1 connector step={step}; no SmolVLM2/DiT/VAE copies.", flush=True)


def main(argv=None):
    args = parse_args(argv)
    # Fail before touching CUDA on the login node.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU training must run inside srun/sbatch")
    if args.workers and len(os.fsencode(tempfile.gettempdir())) > 60:
        raise ValueError("TMPDIR is too long for DataLoader UNIX sockets; use a short path such as /share_4/users/$USER/tmp")
    ctx = setup_distributed()
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("This trainer requires a SLURM GPU allocation")
        with output_lock(args.output_dir) if ctx.is_main else nullcontext():
            train(args, ctx)
    finally:
        cleanup_distributed()


def train(args, ctx):
    barrier()
    if ctx.is_main:
        verify_release(args.data_root)
        if (args.output_dir / "run_contract.json").exists() and args.resume is None:
            raise ValueError("Output already contains a run; specify --resume or a new output directory")
    barrier()
    torch.manual_seed(args.seed)
    cfg, encoder, connector, dit, vae, scheduler = load_stack(args, ctx.device)
    frozen = dict(smolvlm2=encoder, dit=dit, vae=vae)
    if any(p.requires_grad for module in frozen.values() for p in module.parameters()):
        raise RuntimeError("Stage-1 must freeze SmolVLM2, DiT and VAE")
    before = frozen_signatures(frozen)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
    contract = dict(format=FORMAT, data_summary_sha256=sha256_file(args.data_root / "stage1_summary.json"),
                    config_sha256=sha256_file(Path(args.config)), processor=args.processor,
                    processor_sha256=encoder.processor_sha256,
                    frozen_weights_sha256=before,
                    smolvlm2_sha256=sha256_file(Path(cfg.bridge.smolvlm2_ckpt_path)),
                    steps=args.steps, accumulation=args.accumulation, world_size=ctx.world_size,
                    lr=args.lr, warmup=args.warmup, condition_dropout=args.condition_dropout,
                    short_side=args.short_side, long_side=args.long_side, max_tokens=args.max_tokens, seed=args.seed,
                    geometry="nearest_aspect_bucket_resize_full_view_no_crop",
                    temporal="49_distinct_frames_over_whole_source_clip; normalized_duration_not_native_fps",
                    objective="pyramid_flow_mse_only", backbone="released_multistep_t2v_frozen",
                    text_contract="processor_chat_no_modifier_no_128_token_selection",
                    reconstruction="image_only_MLLM_input; no_DiT_clean_target_condition",
                    video_units="all_including_first; uniform_unit_stage_sampling; teacher_forced_causal_history",
                    task_ratio=[1, 1, 1], initialization="random_connector", optimizer="AdamW_fp32_master_eps1e-8")
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT or saved["contract"] != contract:
            raise ValueError("Resume contract mismatch; do not mix data, precision, world size or recipes")
        connector.load_state_dict(saved["connector"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        if not 0 <= start < args.steps:
            raise ValueError("Resume checkpoint already complete or invalid")
        del saved
    end = args.stop_after if args.stop_after is not None else args.steps
    if end <= start:
        raise ValueError("Stop-after must be later than the resumed step")
    model = DDP(connector, device_ids=[ctx.local_rank], broadcast_buffers=False) if ctx.is_distributed else connector
    dataset = Stage1Dataset(args.data_root, steps=args.steps, accumulation=args.accumulation, rank=ctx.rank,
                            world_size=ctx.world_size, seed=args.seed, start_step=start,
                            short_side=args.short_side, long_side=args.long_side)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
                        multiprocessing_context="spawn" if args.workers else None,
                        timeout=120 if args.workers else 0,
                        prefetch_factor=2 if args.workers else None)
    batches = iter(loader)
    if ctx.is_main:
        atomic_json(args.output_dir / "run_contract.json", contract)
        (args.output_dir / "training_complete.json").unlink(missing_ok=True)
        (args.output_dir / "training_paused.json").unlink(missing_ok=True)
        print(f"Stage1: steps={start}->{args.steps} world={ctx.world_size} global_batch={ctx.world_size * args.accumulation} "
              f"trainable_connector={sum(p.numel() for p in connector.parameters()):,} "
              f"tasks=1:1:1 counts={ {k: len(v) for k, v in dataset.records.items()} }", flush=True)
    metrics = validate(args, ctx, encoder, connector, dit, vae, scheduler, dataset)
    if ctx.is_main:
        atomic_json(args.output_dir / f"validation_step{start:06d}.json", metrics)
    started = time.monotonic()
    for step in range(start + 1, end + 1):
        connector.train()
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = {task: [] for task in TASKS}
        observations = []
        for micro in range(args.accumulation):
            sample = next(batches)
            rng = generator_for(ctx.device, args.seed, sample["micro"], ctx.rank)
            layers, mask, latent, dropped = encode_sample(encoder, vae, sample, ctx.device, rng, args.condition_dropout)
            # Independent draws avoid coupling image/video source order to a fixed stage.
            stage = int(torch.randint(3, (), device=ctx.device, generator=rng))
            unit = int(torch.randint(latent.shape[2], (), device=ctx.device, generator=rng))
            sync = model.no_sync() if ctx.is_distributed and micro + 1 != args.accumulation else nullcontext()
            with sync, autocast(ctx.device):
                loss = flow_loss(dit, model, layers, mask, latent, scheduler, stage=stage, unit=unit,
                                 generator=rng, gradient_checkpointing=args.gradient_checkpointing)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Non-finite loss at step={step} task={sample['task']} id={sample['sample_id']}")
                (loss / args.accumulation).backward()
            losses[sample["task"]].append(loss.detach())
            observations.append(dict(task=sample["task"], unit=unit, stage=stage, dropped=dropped,
                                     tokens=mask.shape[1], sample_id=sample["sample_id"],
                                     source_duration_seconds=sample.get("source_duration_seconds"),
                                     effective_sample_fps=sample.get("effective_sample_fps")))
        norm = torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0, error_if_nonfinite=True)
        if not bool(norm > 0):
            raise RuntimeError("Zero connector gradient: frozen DiT must still backpropagate to conditioning")
        if any(p.grad is not None for module in frozen.values() for p in module.parameters()):
            raise RuntimeError("Frozen backbone received gradients")
        optimizer.step()
        if step == start + 1 or step % args.log_every == 0 or step == end:
            report = dict(step=step, lr=lr, grad_norm=scalar_mean(norm, ctx),
                          loss={task: scalar_mean(torch.stack(values).mean(), ctx) for task, values in losses.items()},
                          elapsed_seconds=time.monotonic() - started, rank0_samples=observations,
                          peak_memory_gib=torch.cuda.max_memory_allocated(ctx.device) / 1024**3)
            if ctx.is_main:
                print(json.dumps(report), flush=True)
                with (args.output_dir / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
        if step % args.validate_every == 0 or step == end:
            metrics = validate(args, ctx, encoder, connector, dit, vae, scheduler, dataset)
            if ctx.is_main:
                atomic_json(args.output_dir / f"validation_step{step:06d}.json", metrics)
                print(f"Validation step={step}: {json.dumps(metrics)}", flush=True)
        if step % args.save_every == 0 or step % args.archive_every == 0 or step == end:
            barrier()
            if ctx.is_main:
                save(args, step, connector, optimizer, contract, final=step == args.steps)
            barrier()
    if args.verify_frozen:
        after = frozen_signatures(frozen)
        if before != after:
            raise RuntimeError("Frozen backbone weights changed")
        if ctx.is_main:
            atomic_json(args.output_dir / "frozen_weight_audit.json", dict(before=before, after=after, passed=True))
    if ctx.is_main:
        complete = end == args.steps
        atomic_json(args.output_dir / ("training_complete.json" if complete else "training_paused.json"),
                    dict(step=end, format=FORMAT, frozen_hash_verified=args.verify_frozen,
                         quality_claim="none; evaluate generated samples"))
        print(f"{'Completed' if complete else 'Paused'} Stage-1 alignment at optimizer step {end}.", flush=True)


if __name__ == "__main__":
    main()
