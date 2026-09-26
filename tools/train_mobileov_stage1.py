#!/usr/bin/env python
"""UniVideo-inspired alignment and subsequent T2I/T2V adaptation for NeoDragon.

This is not an exact UniVideo reproduction: public data, MCP connector, pooled
head, native pyramid schedule, equal task weights, and 1/49-frame targets.
Phase 1 freezes the DiT. Phase 2 updates connector + DiT from Phase-1 weights.
Neither phase uses a native text teacher, DreamLite, or DMD.
Phase 2 can optionally add paired T2I image reconstruction as an auxiliary loss.
The opt-in media-epoch recipe also trains text+first-image video conditions,
both with and without the observed frame's VAE latent as an anchor.
UniVideo section 2.4 includes T2V but Table 7 lists one frame for Stage 1;
including video here follows our explicit three-task scope, not that table.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
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
from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training.stage1_vae import encode_posterior
from new_mobile_ov.training import stage2_media_epochs as media_epochs

FORMAT = "mobileov_stage1_mcp_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--init-connector", type=Path, help="Stage-1 inference checkpoint, only for Phase 2")
    parser.add_argument("--expected-init-step", type=int, default=15000)
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--data-root", type=Path, default=Path("download_data/data/univideo_stage1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--steps", type=int, help="Total optimizer updates, not microbatches")
    parser.add_argument("--media-epochs", type=int, choices=(3,),
                        help="Fresh Stage 2: each image/video three times; rotate T2V/reference/anchor modes")
    parser.add_argument("--min-train-videos", type=int, default=1)
    parser.add_argument("--require-expanded-release", action="store_true")
    parser.add_argument("--accumulation", type=int, help="Complete task cycles; Phase 1: 3 tasks, Phase 2: 2")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0,
                        help="Phase-2 auxiliary image flow loss on each T2I target; 0 preserves the text-only recipe")
    parser.add_argument("--enable-reconstruction-on-resume", action="store_true",
                        help="Explicitly fork a text-only Stage-2 checkpoint into the reconstruction recipe")
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--archive-every", type=int)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=2, help="Per task per rank; 3 stages, first/last video units")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--extend-steps", action="store_true",
                        help="Allow only an increase in the resume contract's total steps; preserve the recipe")
    parser.add_argument("--expected-resume-step", type=int,
                        help="Fail if the resume checkpoint is not at this optimizer step")
    parser.add_argument("--stop-after", type=int, help="Pause at this optimizer step to test exact resume; not a new recipe")
    parser.add_argument("--max-runtime-seconds", type=float, default=0,
                        help="Gracefully pause at an update boundary; 0 disables the time budget")
    parser.add_argument("--verify-frozen", action="store_true", help="Full hashes before/after; intended for local smoke")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-window-size", type=int,
                        help="Native causal encoding chunk: Phase 1 defaults to 16, Phase 2 to 8; no frame dropping")
    parser.add_argument("--cuda-memory-limit-gib", type=float, default=0,
                        help="Local memory regression test: cap the PyTorch allocator; 0 leaves it unrestricted")
    args = parser.parse_args(argv)
    if args.media_epochs:
        if args.phase != 2 or args.steps is not None or args.extend_steps or args.enable_reconstruction_on_resume:
            parser.error("Media epochs require Phase 2, auto-calculated steps, and only exact own-run resumes")
        if args.reconstruction_weight <= 0:
            parser.error("Media epochs include paired reconstruction; set --reconstruction-weight (e.g. 0.1)")
    elif args.require_expanded_release or args.min_train_videos != 1:
        parser.error("Expanded-data guards require --media-epochs")
    if args.min_train_videos < 1:
        parser.error("Minimum training video count must be positive")
    args.tasks = TASKS if args.phase == 1 else stage2.TASKS
    if args.steps is None:
        args.steps = 15000 if args.phase == 1 else 100000
    if args.save_every is None:
        args.save_every = 1000 if args.phase == 1 else 10000
    if args.archive_every is None:
        args.archive_every = 5000 if args.phase == 1 else 10000
    if args.accumulation is None:
        args.accumulation = len(args.tasks)
    if args.lr is None:
        args.lr = 1e-4 if args.phase == 1 else 2e-5
    if args.vae_window_size is None:
        args.vae_window_size = 16 if args.phase == 1 else 8
    if args.vae_window_size < 8 or args.vae_window_size % 8:
        parser.error("VAE window must be a positive multiple of 8")
    if not math.isfinite(args.cuda_memory_limit_gib) or args.cuda_memory_limit_gib < 0:
        parser.error("Invalid CUDA memory limit")
    if (args.phase == 2) != (args.init_connector is not None):
        parser.error("Phase 2 requires --init-connector; Phase 1 always starts randomly or resumes its own run")
    if args.max_runtime_seconds < 0 or args.expected_init_step < 1:
        parser.error("Invalid runtime limit or initialization step")
    if not math.isfinite(args.reconstruction_weight) or args.reconstruction_weight < 0:
        parser.error("Reconstruction weight must be finite and nonnegative")
    if args.reconstruction_weight and args.phase != 2:
        parser.error("Auxiliary reconstruction is only for Phase 2; Phase 1 already has its reconstruction task")
    if args.enable_reconstruction_on_resume and (args.resume is None or args.reconstruction_weight == 0):
        parser.error("--enable-reconstruction-on-resume requires --resume and a positive reconstruction weight")
    if (args.extend_steps or args.expected_resume_step is not None) and args.resume is None:
        parser.error("--extend-steps and --expected-resume-step require --resume")
    if args.expected_resume_step is not None and (args.expected_resume_step < 0 or
                                                (not args.media_epochs and args.expected_resume_step >= args.steps)):
        parser.error("Expected resume step must be nonnegative and earlier than the target")
    if min(args.steps, args.accumulation, args.save_every, args.archive_every,
           args.validate_every, args.validation_samples, args.log_every, args.max_tokens) < 1:
        parser.error("Counts must be positive")
    if (not args.media_epochs and args.accumulation % len(args.tasks)) or args.workers < 0 or args.warmup < 0:
        parser.error("Accumulation must contain complete task cycles; workers/warmup cannot be negative")
    if args.stop_after is not None and (args.stop_after < 1 or (not args.media_epochs and args.stop_after > args.steps)):
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


def encode_sample(encoder, vae, sample, device, generator, dropout, *, window_size=16):
    drop = bool(torch.rand((), device=device, generator=generator) < dropout)
    mode = sample.get("mode", sample["task"])
    kwargs = {"allow_text_image": True} if mode in ("i2v_anchor", "i2v_reference") else {}
    layers, mask = encoder(sample["prompt"], sample["image"], drop_condition=drop, **kwargs)
    with torch.no_grad(), autocast(device):
        video = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        posterior = encode_posterior(vae, video, window_size=window_size)
        latent = scale_vae_latents(posterior.sample(generator=generator)).to(torch.bfloat16)
        if mode == "i2v_anchor":
            # Match inference: independently encode ONLY the observed first RGB frame.
            latent[:, :, :1] = encode_anchor(vae, video, generator, window_size=window_size)
    expected = 1 if sample["task"] != "t2v" else 7
    if latent.shape[2] != expected or not bool(torch.isfinite(latent).all()):
        raise ValueError(f"Invalid VAE latent for {sample['task']}: {latent.shape}")
    return layers, mask, latent, drop


@torch.no_grad()
def encode_anchor(vae, video, generator, *, window_size=8):
    with autocast(video.device):
        posterior = encode_posterior(vae, video[:, :, :1], window_size=window_size)
        anchor = scale_vae_latents(posterior.sample(generator=generator)).to(torch.bfloat16)
    if anchor.shape[2] != 1 or not bool(torch.isfinite(anchor).all()):
        raise ValueError("Invalid first-frame VAE anchor")
    return anchor


def generator_for(device, seed, ordinal, rank):
    return torch.Generator(device=device).manual_seed(seed + 1000003 * ordinal + 9176 * rank)


def validate(args, ctx, encoder, connector, dit, vae, scheduler, dataset):
    metrics = {}
    connector.eval()
    was_training = dit.training
    dit.eval()
    with torch.no_grad():
        tasks = (*TASKS, "i2v_reference", "i2v_anchor") if args.media_epochs else TASKS
        for task in tasks:
            source = "t2v" if task in media_epochs.VIDEO_MODES else task
            records = JsonlRecords(args.data_root / "validation" / f"{source}.jsonl")
            values = {}
            for item in range(args.validation_samples):
                index = (item * ctx.world_size + ctx.rank) % len(records)
                sample = prepare_sample(records[index], dataset.sources, args.short_side, args.long_side)
                if source == "t2v" and args.media_epochs:
                    sample = media_epochs.condition_video(sample, task)
                rng = generator_for(ctx.device, args.seed + 888888, index, 0)
                layers, mask, latent, _ = encode_sample(encoder, vae, sample, ctx.device, rng, 0,
                                                       window_size=args.vae_window_size)
                units = [0] if source != "t2v" else [1 if task == "i2v_anchor" else 0, latent.shape[2] - 1]
                for stage in range(3):
                    for unit in units:
                        with autocast(ctx.device):
                            loss = flow_loss(dit, connector, layers, mask, latent, scheduler,
                                             stage=stage, unit=unit, generator=rng, gradient_checkpointing=False)
                        unit_name = "first_future" if task == "i2v_anchor" and unit == 1 else ("first" if unit == 0 else "last")
                        key = f"{task}.stage{stage}.{unit_name}"
                        values.setdefault(key, []).append(loss)
            for key, losses in values.items():
                value = scalar_mean(torch.stack(losses).mean(), ctx)
                if not torch.isfinite(torch.tensor(value)):
                    raise RuntimeError(f"Non-finite validation loss: {key}")
                metrics[key] = value
    connector.train()
    dit.train(was_training)
    return metrics


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume(saved, contract, connector_spec, *, extend_steps=False, expected_step=None,
                    enable_reconstruction=False):
    if saved.get("format") != contract["format"]:
        raise ValueError("Resume checkpoint format mismatch")
    previous = saved.get("contract", {})
    changed = sorted(key for key in set(previous) | set(contract)
                     if previous.get(key) != contract.get(key))
    allowed = set()
    if (extend_steps and isinstance(previous.get("steps"), int)
            and contract["steps"] > previous["steps"]):
        allowed.add("steps")
    aux = contract.get("auxiliary_reconstruction")
    if (enable_reconstruction and contract["format"] == stage2.FORMAT
            and "auxiliary_reconstruction" not in previous and isinstance(aux, dict)
            and aux == stage2.reconstruction_contract(aux.get("weight"))):
        allowed.add("auxiliary_reconstruction")
    if set(changed) - allowed:
        raise ValueError(f"Resume contract mismatch: {changed}. Only an explicit step extension or "
                         "enabling auxiliary reconstruction is allowed; other settings must match")
    step = saved.get("step")
    if type(step) is not int or not 0 <= step <= previous["steps"] or step >= contract["steps"]:
        raise ValueError("Resume checkpoint already complete for this target or has an invalid step")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"Expected resume step={expected_step}, found step={step}")
    if saved.get("connector_spec") != connector_spec:
        raise ValueError("Resume connector architecture mismatch")
    required = ("connector", "dit") if contract["format"] == stage2.FORMAT else ("connector",)
    if any(not saved.get(key) for key in required):
        raise ValueError(f"Resume checkpoint must contain trained weights: {required}")
    optimizer = saved.get("optimizer")
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups") or (step and not optimizer.get("state")):
        raise ValueError("Resume requires optimizer state. Use stage2_resume.pt (or stage1_resume.pt), not an inference checkpoint")
    return step


def save(args, step, connector, optimizer, contract, *, final=False, dit=None):
    payload = dict(format=contract["format"], step=step, connector_spec=connector.spec,
                   connector={key: value.detach().cpu() for key, value in connector.state_dict().items()},
                   contract=contract)
    prefix = "stage1_connector" if dit is None else "stage2_dit_connector"
    if dit is not None:
        payload["dit"] = {key: value.detach().cpu() for key, value in dit.state_dict().items()}
    latest = args.output_dir / f"{prefix}_latest.pt"
    archive = step % args.archive_every == 0 or final
    destination = args.output_dir / f"{prefix}_step{step:06d}.pt" if archive else latest
    atomic_torch_save(payload, destination)
    if archive:
        # Latest and the matching archive share storage, without duplicating a full DiT.
        temporary = latest.with_suffix(".pt.link")
        temporary.unlink(missing_ok=True)
        try:
            os.link(destination, temporary)
        except OSError:
            shutil.copyfile(destination, temporary)
        temporary.replace(latest)
    phase = 1 if dit is None else 2
    atomic_torch_save({**payload, "optimizer": optimizer.state_dict()}, args.output_dir / f"stage{phase}_resume.pt")
    print(f"Saved Stage-{phase} step={step}; optimizer only in resume file; SmolVLM2/VAE excluded.", flush=True)


def runtime_expired(args, ctx, started):
    expired = bool(args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds)
    flag = torch.tensor(int(expired), device=ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag)


def memory_report(ctx):
    if ctx.device.type != "cuda":
        return {}
    keys = ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib")
    local = torch.tensor([torch.cuda.memory_allocated(ctx.device), torch.cuda.memory_reserved(ctx.device),
                          torch.cuda.max_memory_allocated(ctx.device), torch.cuda.max_memory_reserved(ctx.device)],
                         device=ctx.device, dtype=torch.float64) / 1024**3
    rows = [torch.empty_like(local) for _ in range(ctx.world_size)]
    if ctx.is_distributed:
        dist.all_gather(rows, local)
    else:
        rows[0] = local
    return {str(rank): dict(zip(keys, row.tolist())) for rank, row in enumerate(rows)}


@record
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
        if args.cuda_memory_limit_gib:
            capacity = torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
            if args.cuda_memory_limit_gib > capacity:
                raise ValueError("Requested allocator limit exceeds physical GPU capacity")
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_limit_gib / capacity, ctx.device)
            print(f"[rank{ctx.rank}] PyTorch allocator capped at {args.cuda_memory_limit_gib} GiB; "
                  "this does not emulate GPU kernels or cap NCCL allocations", flush=True)
        with output_lock(args.output_dir) if ctx.is_main else nullcontext():
            train(args, ctx)
    finally:
        cleanup_distributed()


def train(args, ctx):
    launched = time.monotonic()
    barrier()
    if ctx.is_main:
        verify_release(args.data_root)
        if (args.output_dir / "run_contract.json").exists() and args.resume is None:
            raise ValueError("Output already contains a run; specify --resume or a new output directory")
    barrier()
    epoch_plan = None
    if args.media_epochs:
        epoch_plan = media_epochs.plan_from_release(
            args.data_root, world_size=ctx.world_size, accumulation=args.accumulation,
            epochs=args.media_epochs, seed=args.seed, min_videos=args.min_train_videos,
            require_expansion=args.require_expanded_release, verify=False)
        args.steps = epoch_plan.steps
        if args.stop_after is not None and args.stop_after > args.steps:
            raise ValueError("Stop-after exceeds calculated epoch budget")
        if ctx.is_main:
            print(f"Exact media epoch plan: {json.dumps(epoch_plan.contract())}", flush=True)
    torch.manual_seed(args.seed)
    cfg, encoder, connector, dit, vae, scheduler = load_stack(args, ctx.device)
    initial_modules = dict(smolvlm2=encoder, dit=dit, vae=vae)
    if any(p.requires_grad for module in initial_modules.values() for p in module.parameters()):
        raise RuntimeError("Stage-1 must freeze SmolVLM2, DiT and VAE")
    initial_signatures = frozen_signatures(initial_modules)
    frozen = initial_modules if args.phase == 1 else dict(smolvlm2=encoder, vae=vae)
    before = {key: initial_signatures[key] for key in frozen}
    contract = dict(format=FORMAT if args.phase == 1 else stage2.FORMAT,
                    data_summary_sha256=sha256_file(args.data_root / "stage1_summary.json"),
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
    # Keep old default Stage-1 resume contracts valid; record nondefault encoding explicitly.
    if args.phase == 2 or args.vae_window_size != 16:
        contract["vae_window_size"] = args.vae_window_size
    if args.phase == 2:
        payload = torch.load(args.init_connector, map_location="cpu", weights_only=False)
        stage2.initialize_from_alignment(
            payload, connector, dit, expected_step=args.expected_init_step, signatures=initial_signatures,
            config_sha256=contract["config_sha256"],
            processor_sha256=encoder.processor_sha256, args=args)
        contract.update(initial_weights_sha256=initial_signatures,
                        initialization="stage1_connector_and_released_multistep_dit",
                        init_connector_sha256=sha256_file(args.init_connector), init_step=payload["step"],
                        alignment_data_sha256=payload["contract"]["data_summary_sha256"],
                        task_ratio=[1, 1], tasks=list(args.tasks), backbone="released_multistep_t2v_trainable",
                        frozen_components=["smolvlm2", "vae"], trainable_components=["connector", "dit"],
                        ema=False, scope="T2I_T2V_public_data_adaptation; not_exact_UniVideo_stage2")
        del payload
    if args.reconstruction_weight:
        contract["auxiliary_reconstruction"] = stage2.reconstruction_contract(args.reconstruction_weight)
    if epoch_plan:
        contract.update(media_epochs=epoch_plan.contract(), task_ratio="source_proportional_exposure; loss_balanced",
                        tasks=list(media_epochs.METRIC_TASKS),
                        scope="fresh_multimodal_stage2; public_data; not_exact_UniVideo_recipe",
                        video_units="T2V/reference:0_to6; anchor:1_to6; teacher_forced_causal_history",
                        text_contract="processor_chat; I2V_text_and_first_image_together; no_truncation")
    trainable = connector if args.phase == 1 else stage2.JointFlowModel(connector, dit)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0, foreach=False)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        start = validate_resume(saved, contract, connector.spec, extend_steps=args.extend_steps,
                                expected_step=args.expected_resume_step,
                                enable_reconstruction=args.enable_reconstruction_on_resume)
        if "auxiliary_reconstruction" in contract and "auxiliary_reconstruction" not in saved["contract"]:
            if args.output_dir.resolve() == args.resume.resolve().parent or (args.output_dir / "run_contract.json").exists():
                raise ValueError("Enabling reconstruction requires a new output directory; preserve the text-only baseline")
        connector.load_state_dict(saved["connector"], strict=True)
        if args.phase == 2:
            dit.load_state_dict(saved["dit"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if ctx.is_main:
            atomic_json(args.output_dir / f"resume_from_step{start:06d}.json",
                        dict(checkpoint=str(args.resume.resolve()), checkpoint_step=start,
                             previous_target_steps=saved["contract"]["steps"], target_steps=args.steps,
                             optimizer_restored=True, previous_contract=saved["contract"],
                             changed_contract_fields=sorted(key for key in set(saved["contract"]) | set(contract)
                                                           if saved["contract"].get(key) != contract.get(key))))
            print(f"Resumed optimizer + trained weights at step={start}; target={args.steps}; "
                  "sample sequence and warmup continue from the absolute step.", flush=True)
        del saved
    end = args.stop_after if args.stop_after is not None else args.steps
    if end <= start:
        raise ValueError("Stop-after must be later than the resumed step")
    trainable_modules = dict(connector=connector)
    if args.phase == 2:
        trainable_modules["dit"] = dit
    trainable_before = frozen_signatures(trainable_modules) if args.verify_frozen else None
    model = DDP(trainable, device_ids=[ctx.local_rank], broadcast_buffers=False,
                gradient_as_bucket_view=True, find_unused_parameters=args.phase == 2) if ctx.is_distributed else trainable
    if epoch_plan:
        dataset = media_epochs.MediaEpochDataset(args.data_root, plan=epoch_plan, rank=ctx.rank,
                                                start_step=start, short_side=args.short_side, long_side=args.long_side)
    else:
        dataset = Stage1Dataset(args.data_root, steps=args.steps, accumulation=args.accumulation, rank=ctx.rank,
                                world_size=ctx.world_size, seed=args.seed, start_step=start,
                                short_side=args.short_side, long_side=args.long_side, tasks=args.tasks)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
                        multiprocessing_context="spawn" if args.workers else None,
                        timeout=120 if args.workers else 0,
                        prefetch_factor=2 if args.workers else None)
    batches = iter(loader)
    if ctx.is_main:
        atomic_json(args.output_dir / "run_contract.json", contract)
        (args.output_dir / "training_complete.json").unlink(missing_ok=True)
        (args.output_dir / "training_paused.json").unlink(missing_ok=True)
        print(f"Stage{args.phase}: steps={start}->{args.steps} world={ctx.world_size} global_batch={ctx.world_size * args.accumulation} "
              f"trainable_connector={sum(p.numel() for p in connector.parameters()):,} "
              f"trainable_dit={sum(p.numel() for p in dit.parameters() if p.requires_grad):,} "
              f"tasks={contract.get('tasks', args.tasks)} counts={ {k: len(v) for k, v in dataset.records.items()} } "
              f"vae_window_size={args.vae_window_size}", flush=True)
        if args.reconstruction_weight:
            print(f"Auxiliary reconstruction: L=0.5*mean(T2I)+0.5*mean(video modes)+{args.reconstruction_weight}*mean(image_reconstruction); "
                  "paired T2I image, no caption/clean-latent condition, no extra VAE encoding.", flush=True)
    metrics = validate(args, ctx, encoder, connector, dit, vae, scheduler, dataset)
    if ctx.is_main:
        atomic_json(args.output_dir / f"validation_step{start:06d}.json", metrics)
    started = time.monotonic()
    for step in range(start + 1, end + 1):
        trainable.train()
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = {task: [] for task in (media_epochs.METRIC_TASKS if epoch_plan else args.tasks)}
        weighted_loss = torch.zeros((), device=ctx.device)
        if args.reconstruction_weight:
            losses["image_reconstruction"] = []
        observations = []
        for micro in range(args.accumulation):
            sample = next(batches)
            rng = generator_for(ctx.device, args.seed, sample["micro"], ctx.rank)
            layers, mask, latent, dropped = encode_sample(encoder, vae, sample, ctx.device, rng, args.condition_dropout,
                                                         window_size=args.vae_window_size)
            # Independent draws avoid coupling image/video source order to a fixed stage.
            stage = int(torch.randint(3, (), device=ctx.device, generator=rng))
            mode = sample.get("mode", sample["task"])
            unit = media_epochs.sample_unit(mode, latent.shape[2], ctx.device, rng)
            scale = sample.get("loss_scale", 1.0)
            padding = sample.get("padding", False)
            sync = model.no_sync() if ctx.is_distributed and micro + 1 != args.accumulation else nullcontext()
            with sync, autocast(ctx.device):
                if args.phase == 1:
                    loss = flow_loss(dit, model, layers, mask, latent, scheduler, stage=stage, unit=unit,
                                     generator=rng, gradient_checkpointing=args.gradient_checkpointing)
                else:
                    loss = model(layers, mask, latent, scheduler, stage=stage, unit=unit, generator=rng)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Non-finite loss at step={step} task={sample['task']} id={sample['sample_id']}")
                (loss * scale / args.accumulation).backward()
                weighted_loss += loss.detach() * scale / args.accumulation
                if args.reconstruction_weight and sample["task"] == "t2i":
                    # Separate RNG and backward pass preserve text sampling and bound activation memory.
                    rec_rng = generator_for(ctx.device, args.seed + stage2.RECONSTRUCTION_SEED_OFFSET,
                                            sample["micro"], ctx.rank)
                    rec_layers, rec_mask = stage2.reconstruction_condition(encoder, sample)
                    rec_stage = int(torch.randint(3, (), device=ctx.device, generator=rec_rng))
                    rec_loss = model(rec_layers, rec_mask, latent, scheduler, stage=rec_stage, unit=0, generator=rec_rng)
                    if not bool(torch.isfinite(rec_loss)):
                        raise RuntimeError(f"Non-finite reconstruction loss at step={step} id={sample['sample_id']}")
                    rec_scale = (2 * args.reconstruction_weight * scale / args.accumulation if epoch_plan else
                                 stage2.reconstruction_coefficient(args.reconstruction_weight, args.accumulation))
                    (rec_loss * rec_scale).backward()
                    weighted_loss += rec_loss.detach() * rec_scale
                    if not padding:
                        losses["image_reconstruction"].append(rec_loss.detach())
                    observations.append(dict(task="image_reconstruction", paired_with="t2i", unit=0,
                                             stage=rec_stage, dropped=False, tokens=rec_mask.shape[1],
                                             sample_id=sample["sample_id"], **({"padding": padding} if epoch_plan else {})))
                    del rec_layers, rec_mask, rec_loss
            if not padding:
                losses[mode].append(loss.detach())
            observations.append(dict(task=mode, unit=unit, stage=stage, dropped=dropped,
                                     tokens=mask.shape[1], sample_id=sample["sample_id"],
                                     source_duration_seconds=sample.get("source_duration_seconds"),
                                     effective_sample_fps=sample.get("effective_sample_fps"),
                                     **({"epoch": sample["epoch"] + 1, "padding": padding,
                                         "loss_scale": scale} if epoch_plan else {})))
            del layers, mask, latent, loss
        component_norms = dict(connector=stage2.gradient_norm(connector))
        if args.phase == 2:
            component_norms["dit"] = stage2.gradient_norm(dit)
        norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0, error_if_nonfinite=True)
        if any(not bool(value > 0) for value in component_norms.values()):
            raise RuntimeError("A trainable component received zero gradient")
        if any(p.grad is not None for module in frozen.values() for p in module.parameters()):
            raise RuntimeError("Frozen backbone received gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if runtime_expired(args, ctx, launched):
            end = step
        if step == start + 1 or step % args.log_every == 0 or step == end:
            loss_report, counts = media_epochs.distributed_loss_means(losses, ctx) if epoch_plan else (
                {task: scalar_mean(torch.stack(values).mean(), ctx) for task, values in losses.items()}, None)
            report = dict(step=step, lr=lr, grad_norm=scalar_mean(norm, ctx),
                          component_grad_norm={key: scalar_mean(value, ctx) for key, value in component_norms.items()},
                          loss=loss_report,
                          elapsed_seconds=time.monotonic() - started, rank0_samples=observations,
                          peak_memory_gib=torch.cuda.max_memory_allocated(ctx.device) / 1024**3,
                          memory_by_rank=memory_report(ctx))
            if epoch_plan:
                report.update(epoch=(step - 1) // epoch_plan.steps_per_epoch + 1,
                              steps_per_epoch=epoch_plan.steps_per_epoch,
                              valid_samples_in_update=counts, weighted_loss=scalar_mean(weighted_loss, ctx))
            elif args.reconstruction_weight:
                report["loss_weights"] = dict(t2i=0.5, t2v=0.5, image_reconstruction=args.reconstruction_weight)
                report["weighted_loss"] = sum(report["loss"][task] * weight
                                              for task, weight in report["loss_weights"].items())
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
                save(args, step, connector, optimizer, contract, final=step == args.steps,
                     dit=dit if args.phase == 2 else None)
            barrier()
        if step == end:
            break
    memory = memory_report(ctx)
    if ctx.is_main:
        atomic_json(args.output_dir / "memory_summary.json",
                    dict(allocator_limit_gib=args.cuda_memory_limit_gib, by_rank=memory))
    if args.verify_frozen:
        after = frozen_signatures(frozen)
        if before != after:
            raise RuntimeError("Frozen backbone weights changed")
        trainable_after = frozen_signatures(trainable_modules)
        unchanged = [key for key in trainable_modules if trainable_before[key] == trainable_after[key]]
        if unchanged:
            raise RuntimeError(f"Trainable weights did not change: {unchanged}")
        if ctx.is_main:
            atomic_json(args.output_dir / "frozen_weight_audit.json",
                        dict(before=before, after=after, passed=True,
                             trainable_before=trainable_before, trainable_after=trainable_after))
    if ctx.is_main:
        complete = end == args.steps
        atomic_json(args.output_dir / ("training_complete.json" if complete else "training_paused.json"),
                    dict(step=end, format=contract["format"], frozen_hash_verified=args.verify_frozen,
                         quality_claim="none; evaluate generated samples"))
        print(f"{'Completed' if complete else 'Paused'} Stage-{args.phase} at optimizer step {end}.", flush=True)


if __name__ == "__main__":
    main()
