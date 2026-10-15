#!/usr/bin/env python
"""Job B: the Job-A generation fork plus LLM LoRA and real answer-only QA replay."""
import argparse
from contextlib import nullcontext
import json
import math
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record

from new_mobile_ov.training import job_b_model as job
from new_mobile_ov.training import job_b_qa as qa
from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training.stage1_alignment import frozen_signatures, JsonlRecords
from new_mobile_ov.training.stage1_vae import encode_posterior
from tools import train_mobileov_stage1 as base
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file, output_lock


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--qa-every", type=int, default=8)
    parser.add_argument("--qa-weight", type=float, default=.1)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--lora-warmup", type=int, default=1000)
    extra, remaining = parser.parse_known_args(argv)
    args = base.parse_args(remaining)
    if not args.init_stage2 or not math.isfinite(extra.qa_weight) or not 0 < extra.qa_weight <= 1:
        parser.error("Job B requires the weights-only epoch-3 fork and positive QA CE")
    if extra.qa_every < 1 or extra.lora_warmup < 1 or not math.isfinite(extra.lora_lr) or not 0 < extra.lora_lr <= args.lr:
        parser.error("Invalid LoRA LR, replay interval or warmup")
    for key, value in vars(extra).items():
        setattr(args, key, value)
    return args


@torch.no_grad()
def target_latent(vae, sample, device, rng, window_size):
    with base.autocast(device):
        video = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        latent = base.scale_vae_latents(encode_posterior(vae, video, window_size=window_size).sample(generator=rng))
        latent = latent.to(torch.bfloat16)
        if sample.get("mode") == "i2v_anchor":
            latent[:, :, :1] = base.encode_anchor(vae, video, rng, window_size=window_size)
    if latent.shape[2] != (7 if sample["task"] == "t2v" else 1) or not bool(torch.isfinite(latent).all()):
        raise ValueError("Invalid Job-B VAE target")
    return latent


def optimizer_for(owner, args):
    adapters = [p for p in owner.encoder.parameters() if p.requires_grad]
    parameters = list(owner.connector.parameters()) + list(owner.dit.parameters())
    if not adapters or any(p.dtype != torch.float32 for p in [*adapters, *parameters]):
        raise ValueError("Job B requires FP32 master weights, including LoRA")
    return torch.optim.AdamW([
        dict(params=parameters, lr=args.lr, base_lr=args.lr, warmup=args.warmup, name="generation"),
        dict(params=adapters, lr=args.lora_lr, base_lr=args.lora_lr, warmup=args.lora_warmup, name="llm_lora")],
        betas=(.9, .95), eps=1e-8, weight_decay=0, foreach=False)


def restore(payload, owner, optimizer, contract, ctx):
    if payload.get("format") != job.FORMAT or payload.get("contract") != contract:
        raise ValueError("Job-B resume contract differs; do not resume Job A or an old generation-only run")
    step = payload["step"]
    if not contract["init_step"] < step < contract["steps"]:
        raise ValueError("Invalid or already-completed Job-B resume step")
    if not payload.get("optimizer", {}).get("state") or len(payload.get("rng_states", [])) != ctx.world_size:
        raise ValueError("Use job_b_resume.pt with optimizer and per-rank RNG, not inference weights")
    owner.connector.load_state_dict(payload["connector"], strict=True)
    owner.dit.load_state_dict(payload["dit"], strict=True)
    job.load_lora(owner.encoder, payload["smolvlm2_lora"])
    optimizer.load_state_dict(payload["optimizer"])
    return step, payload["rng_states"]


def save(args, step, owner, optimizer, contract, rng_states, final=False):
    payload = dict(format=job.FORMAT, step=step, contract=contract, connector_spec=owner.connector.spec,
                   lora_spec=owner.encoder.lora_spec, smolvlm2_lora=job.lora_state(owner.encoder),
                   connector={key: value.detach().cpu() for key, value in owner.connector.state_dict().items()},
                   dit={key: value.detach().cpu() for key, value in owner.dit.state_dict().items()})
    latest = args.output_dir / "job_b_dit_connector_lora_latest.pt"
    archived = final or (step - contract["init_step"]) % args.archive_every == 0
    destination = args.output_dir / f"job_b_dit_connector_lora_step{step:06d}.pt" if archived else latest
    base.atomic_torch_save(payload, destination)
    if archived:
        temporary = latest.with_suffix(".link")
        temporary.unlink(missing_ok=True)
        try:
            os.link(destination, temporary)
        except OSError:
            shutil.copyfile(destination, temporary)
        temporary.replace(latest)
    base.atomic_torch_save(dict(payload, optimizer=optimizer.state_dict(), rng_states=rng_states),
                           args.output_dir / "job_b_resume.pt")
    print(f"Saved Job B step={step}; inference contains LoRA+MCP+DiT, optimizer only in resume.", flush=True)


def validate(args, ctx, owner, vae, scheduler, dataset):
    state, was_training = base.capture_rng_states(ctx), owner.training
    owner.eval()
    try:
        values = base.validate(args, ctx, owner.encoder, owner.connector, owner.dit, vae, scheduler, dataset)
        with torch.no_grad(), base.autocast(ctx.device):
            for modality in qa.MODALITIES:
                records = JsonlRecords(args.qa_root / "validation" / f"{modality}.jsonl")
                losses = []
                for index in range(args.validation_samples):
                    row = records[(index * ctx.world_size + ctx.rank) % len(records)]
                    batch = qa.pack_answer(owner.encoder.processor, row, qa.read_images(args.qa_root, row), args.max_tokens)
                    losses.append(owner.encoder.qa_loss(batch))
                values[f"qa_answer_ce.{modality}"] = base.scalar_mean(torch.stack(losses).mean(), ctx)
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Nonfinite Job-B validation metric")
        return values
    finally:
        owner.train(was_training)
        base.restore_rng_states(state, ctx)


def train(args, ctx):
    launched = time.monotonic()
    if ctx.is_main:
        if (args.output_dir / "run_contract.json").exists() and args.resume is None:
            raise ValueError("Job-B output already contains a run; resume explicitly or use a new OUT")
    base.barrier()
    qa_report = qa.verify_release(args.qa_root)
    if qa_report.get("integration_smoke") and not args.integration_smoke:
        raise ValueError("Synthetic smoke QA is not a production understanding dataset")
    if not args.integration_smoke and any(qa_report["files"][f"train/{modality}.jsonl"]["rows"] < 100
                                         or qa_report["files"][f"validation/{modality}.jsonl"]["rows"] < 10
                                         for modality in qa.MODALITIES):
        raise ValueError("Unexpectedly small QA release; require >=100 train and >=10 validation rows per modality")
    plan = media.plan_from_release(args.data_root, world_size=ctx.world_size, accumulation=args.accumulation,
                                   epochs=1, start_epoch=3, seed=args.seed, min_videos=args.min_train_videos,
                                   require_expansion=args.require_expanded_release)
    args.steps = args.expected_stage2_step + plan.steps
    cfg, frozen_encoder, connector, dit, vae, scheduler = base.load_stack(args, ctx.device)
    signatures = frozen_signatures(dict(smolvlm2=frozen_encoder, vae=vae))
    parent_sha = [sha256_file(args.init_stage2) if ctx.is_main else None]
    if ctx.is_distributed:
        dist.broadcast_object_list(parent_sha, src=0)
    if args.expected_stage2_sha256 and parent_sha[0] != args.expected_stage2_sha256:
        raise ValueError("Job-B epoch-3 parent digest differs")
    if args.output_dir.resolve() == args.init_stage2.resolve().parent:
        raise ValueError("Preserve the epoch-3 parent in its original output directory")
    original = dict(format=stage2.FORMAT, config_sha256=sha256_file(Path(args.config)), processor=args.processor,
                    processor_sha256=frozen_encoder.processor_sha256,
                    frozen_weights_sha256=dict(smolvlm2=signatures["smolvlm2"], vae=signatures["vae"]),
                    short_side=args.short_side, long_side=args.long_side, max_tokens=args.max_tokens,
                    condition_dropout=args.condition_dropout, vae_window_size=args.vae_window_size,
                    geometry="nearest_aspect_bucket_resize_full_view_no_crop",
                    temporal="49_distinct_frames_over_whole_source_clip; normalized_duration_not_native_fps")
    if qa_report["processor_sha256"] != frozen_encoder.processor_sha256 or qa_report["max_tokens"] != args.max_tokens:
        raise ValueError("Prepared QA processor/token budget differs from the generation encoder")
    parent = torch.load(args.init_stage2, map_location="cpu", weights_only=False, mmap=True)
    media.validate_fork_parent(parent, plan, expected_step=args.expected_stage2_step,
                              data_summary_sha256=sha256_file(args.data_root / "stage1_summary.json"),
                              integration_smoke=args.integration_smoke)
    stage2.initialize_from_stage2(parent, connector, dit, contract=original, args=args)
    encoder = job.LoRAEncoder(frozen_encoder, gradient_checkpointing=args.gradient_checkpointing)
    owner = job.JointModel(encoder, connector, dit)
    contract = dict(original, format=job.FORMAT, init_step=args.expected_stage2_step,
                    init_stage2_sha256=parent_sha[0], parent_contract=parent["contract"], steps=args.steps,
                    media_epochs=plan.contract(), data_summary_sha256=sha256_file(args.data_root / "stage1_summary.json"),
                    qa_summary_sha256=sha256_file(args.qa_root / "qa_summary.json"), seed=args.seed,
                    generation_lr=args.lr, generation_warmup=args.warmup,
                    lora_lr=args.lora_lr, lora_warmup=args.lora_warmup, lora_spec=encoder.lora_spec,
                    qa_every=args.qa_every, qa_weight=args.qa_weight, qa_ratio=[.5, .2, .3],
                    qa_presentations=(plan.steps // args.qa_every) * ctx.world_size,
                    qa_counts={name: info["rows"] for name, info in qa_report["files"].items()},
                    optimizer=dict(name="AdamW_fp32_master", betas=[.9, .95], eps=1e-8, weight_decay=0),
                    reconstruction_weight=args.reconstruction_weight, t2v_first_probability=.5,
                    objective="same_generation_as_Job_A_plus_answer_only_CE_before_each_replay_update",
                    kl=False, ema=False, history_noise=0,
                    frozen_components=["VLM_base", "vision", "visual_projector", "embedding", "LM_head", "VAE"],
                    trainable_components=["LLM_LoRA", "MCP", "DiT"],
                    integration_smoke=args.integration_smoke,
                    scope="public_QA_replay; not_exact_Mobile_O_or_UniVideo; no_quality_or_forgetting_guarantee")
    del parent, frozen_encoder
    optimizer = optimizer_for(owner, args)
    start, resume_rng = args.expected_stage2_step, None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        start, resume_rng = restore(payload, owner, optimizer, contract, ctx)
        if args.expected_resume_step is not None and start != args.expected_resume_step:
            raise ValueError("Job-B resume step differs from --expected-resume-step")
        del payload
    end = args.stop_after or args.steps
    if not start < end <= args.steps:
        raise ValueError("Stop-after must be after the resumed step and within the media budget")
    model = DDP(owner, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=True) if ctx.is_distributed else owner
    frozen_versions = [(p, p._version) for p in [*owner.parameters(), *vae.parameters()] if not p.requires_grad]
    dataset = media.MediaEpochDataset(args.data_root, plan=plan, rank=ctx.rank,
                                     start_step=start - args.expected_stage2_step,
                                     short_side=args.short_side, long_side=args.long_side)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
                        multiprocessing_context="spawn" if args.workers else None,
                        prefetch_factor=2 if args.workers else None, timeout=120 if args.workers else 0)
    batches = iter(loader)
    replay = qa.Replay(args.qa_root, processor=encoder.processor, seed=args.seed, rank=ctx.rank,
                       world_size=ctx.world_size, max_tokens=args.max_tokens)
    if ctx.is_main:
        atomic_json(args.output_dir / "run_contract.json", contract)
        (args.output_dir / "training_paused.json").unlink(missing_ok=True)
        (args.output_dir / "training_complete.json").unlink(missing_ok=True)
        print(f"Job B step={start}->{args.steps}; media updates={plan.steps}; QA presentations={contract['qa_presentations']} (not unique).", flush=True)
    metrics = validate(args, ctx, owner, vae, scheduler, dataset)
    if ctx.is_main:
        atomic_json(args.output_dir / f"validation_step{start:06d}.json", metrics)
    if resume_rng is not None:
        base.restore_rng_states(resume_rng, ctx)
    adapter_before = job.lora_state(encoder)
    for step in range(start + 1, end + 1):
        local_step = step - args.expected_stage2_step
        owner.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * min(1, local_step / max(1, group["warmup"]))
        qa_loss, qa_info = None, None
        # QA precedes generation so the final synchronized forward uses ALL trainable components.
        if local_step % args.qa_every == 0:
            batch = replay.sample(local_step // args.qa_every - 1)
            with model.no_sync() if ctx.is_distributed else nullcontext(), base.autocast(ctx.device):
                qa_loss = model(qa=batch)
                if not bool(torch.isfinite(qa_loss)):
                    raise RuntimeError("Nonfinite answer-only QA CE")
                (args.qa_weight * qa_loss).backward()
            qa_info = dict(sample_id=batch["sample_id"], modality=batch["modality"],
                           answer_tokens=int(batch["answer_mask"].sum()))
            qa_info["lora_grad_norm_from_QA_only"] = base.scalar_mean(stage2.gradient_norm(encoder), ctx)
            if not math.isfinite(qa_info["lora_grad_norm_from_QA_only"]) or qa_info["lora_grad_norm_from_QA_only"] <= 0:
                raise RuntimeError("Answer CE did not reach LLM LoRA")
            if ctx.is_main and local_step // args.qa_every == 1:
                atomic_json(args.output_dir / "first_qa_gradient_audit.json", dict(step=step, **qa_info))
            qa_loss = qa_loss.detach()
            del batch
        extra_metrics = ("t2v.first", "t2v.future")
        losses = {key: [] for key in (*media.METRIC_TASKS, *extra_metrics)}
        weighted, observations = torch.zeros((), device=ctx.device), []
        for micro in range(args.accumulation):
            sample = next(batches)
            rng = base.generator_for(ctx.device, args.seed, sample["micro"], ctx.rank)
            dropped = bool(torch.rand((), device=ctx.device, generator=rng) < args.condition_dropout)
            latent = target_latent(vae, sample, ctx.device, rng, args.vae_window_size)
            stage = int(torch.randint(3, (), device=ctx.device, generator=rng))
            mode = sample.get("mode", sample["task"])
            unit = media.sample_unit(mode, latent.shape[2], ctx.device, rng, t2v_first_probability=.5)
            rec = sample["task"] == "t2i"
            final = micro + 1 == args.accumulation
            with model.no_sync() if ctx.is_distributed and (not final or rec) else nullcontext(), base.autocast(ctx.device):
                loss = model(prompt=sample["prompt"], image=sample["image"], drop=dropped,
                             allow_text_image=mode in media.VIDEO_MODES[1:], latent=latent,
                             scheduler=scheduler, stage=stage, unit=unit, generator=rng)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite generation loss")
                (loss * sample["loss_scale"] / args.accumulation).backward()
            weighted += loss.detach() * sample["loss_scale"] / args.accumulation
            if not sample["padding"]:
                losses[mode].append(loss.detach())
                if mode == "t2v":
                    losses["t2v.first" if unit == 0 else "t2v.future"].append(loss.detach())
            observations.append(dict(task=mode, sample_id=sample["sample_id"], stage=stage, unit=unit,
                                     epoch=sample["epoch"] + 1, padding=sample["padding"], dropped=dropped,
                                     image_condition=sample["image"] is not None and not dropped,
                                     loss_scale=sample["loss_scale"]))
            del loss
            if rec:
                rec_rng = base.generator_for(ctx.device, args.seed + stage2.RECONSTRUCTION_SEED_OFFSET,
                                             sample["micro"], ctx.rank)
                rec_stage = int(torch.randint(3, (), device=ctx.device, generator=rec_rng))
                with model.no_sync() if ctx.is_distributed and not final else nullcontext(), base.autocast(ctx.device):
                    loss = model(prompt="", image=media.first_frame_image(sample["video"]), latent=latent,
                                 scheduler=scheduler, stage=rec_stage, unit=0, generator=rec_rng)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite paired reconstruction loss")
                    scale = 2 * args.reconstruction_weight * sample["loss_scale"] / args.accumulation
                    (loss * scale).backward()
                weighted += loss.detach() * scale
                if not sample["padding"]:
                    losses["image_reconstruction"].append(loss.detach())
                del loss
            del latent
        norms = {name: base.scalar_mean(stage2.gradient_norm(module), ctx)
                 for name, module in (("llm_lora", encoder), ("connector", connector), ("dit", dit))}
        if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
            raise RuntimeError(f"A trainable component has no finite global gradient: {norms}")
        if any(p.grad is not None or p._version != version for p, version in frozen_versions):
            raise RuntimeError("Frozen base/VAE parameters received gradients or changed")
        norm = torch.nn.utils.clip_grad_norm_([p for p in owner.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        optimizer.step()
        if step == start + 1:
            changed = any(not torch.equal(adapter_before[key], value) for key, value in job.lora_state(encoder).items())
            if not changed:
                raise RuntimeError("LoRA gradients existed but optimizer did not change adapter weights")
            if ctx.is_main:
                atomic_json(args.output_dir / "first_update_gradient_audit.json",
                            dict(step=step, component_grad_norm=norms, lora_weights_changed=True,
                                 frozen_parameter_versions_unchanged=True,
                                 qa_gradient_audit_pending=not (args.output_dir / "first_qa_gradient_audit.json").exists()))
            del adapter_before
        optimizer.zero_grad(set_to_none=True)
        if base.runtime_expired(args, ctx, launched):
            end = step
        if step == start + 1 or local_step % args.log_every == 0 or step == end or qa_info:
            report, counts = media.distributed_loss_means(losses, ctx, extra_tasks=extra_metrics)
            record = dict(step=step, optimizer_step=local_step, generation_loss=report,
                          weighted_generation_loss=base.scalar_mean(weighted, ctx), valid_samples=counts,
                          qa_answer_ce=base.scalar_mean(qa_loss, ctx) if qa_loss is not None else None,
                          rank0_qa_sample=qa_info, component_grad_norm=norms,
                          rank0_generation_samples=observations, steps_per_epoch=plan.steps_per_epoch,
                          elapsed_seconds=time.monotonic() - launched,
                          grad_norm=base.scalar_mean(norm, ctx), lr={g["name"]: g["lr"] for g in optimizer.param_groups},
                          memory_by_rank=base.memory_report(ctx))
            if ctx.is_main:
                print(json.dumps(record), flush=True)
                with (args.output_dir / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        if local_step % args.validate_every == 0 or step == end:
            metrics = validate(args, ctx, owner, vae, scheduler, dataset)
            if ctx.is_main:
                atomic_json(args.output_dir / f"validation_step{step:06d}.json", metrics)
                print(f"Validation Job B step={step}: {json.dumps(metrics)}", flush=True)
        if local_step % args.save_every == 0 or local_step % args.archive_every == 0 or step == end:
            base.barrier()
            rng_states = base.capture_rng_states(ctx)
            if ctx.is_main:
                save(args, step, owner, optimizer, contract, rng_states, final=step == args.steps)
            base.barrier()
        if step == end:
            break
    if ctx.is_main:
        atomic_json(args.output_dir / ("training_complete.json" if end == args.steps else "training_paused.json"),
                    dict(format=job.FORMAT, step=end, qa_release_sha256=contract["qa_summary_sha256"],
                         quality_claim="none; evaluate generation and understanding vs frozen base"))


@record
def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Job B training requires a Slurm GPU allocation")
    ctx = base.setup_distributed()
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("Job B production training requires CUDA")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        with output_lock(args.output_dir) if ctx.is_main else nullcontext():
            train(args, ctx)
    finally:
        base.cleanup_distributed()


if __name__ == "__main__":
    main()
