#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.bridge import MobileOVNeodragonTextBridge
from new_mobile_ov.checkpoints import ensure_neodragon_assets
from new_mobile_ov.config import load_config
from new_mobile_ov.training.distributed import (
    barrier,
    build_deepspeed_config,
    cleanup_distributed,
    full_state_dict,
    rank0_print,
    scalar_mean,
    setup_distributed,
    write_deepspeed_config,
)


class VideoPromptDataset(Dataset):
    def __init__(self, manifest: str | Path, max_samples: int = -1):
        df = pd.read_csv(manifest)
        video_col = next((c for c in ["video_path", "path", "mp4"] if c in df.columns), None)
        prompt_col = next((c for c in ["prompt", "caption", "text"] if c in df.columns), None)
        if video_col is None:
            raise ValueError(f"{manifest} must contain one of columns: video_path, path, mp4")
        if prompt_col is None:
            raise ValueError(f"{manifest} must contain one of columns: prompt, caption, text")
        df = df[[video_col, prompt_col]].dropna()
        if max_samples > 0:
            df = df.head(max_samples)
        self.video_paths = [str(x) for x in df[video_col].tolist()]
        self.prompts = [str(x) for x in df[prompt_col].tolist()]
        if not self.video_paths:
            raise ValueError(f"No samples found in {manifest}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"video_path": self.video_paths[index], "prompt": self.prompts[index]}


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def read_video_uniform(path: str, *, num_frames: int, height: int, width: int) -> torch.Tensor:
    try:
        import imageio.v3 as iio

        frames = [frame for frame in iio.imiter(path)]
    except Exception:
        import cv2

        cap = cv2.VideoCapture(path)
        frames = []
        ok, frame = cap.read()
        while ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ok, frame = cap.read()
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read video frames from {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    out = []
    for i in idx:
        image = Image.fromarray(frames[int(i)]).convert("RGB").resize((width, height), Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(out, dim=1)  # [C,T,H,W]


def collate_video_batch(batch: list[dict[str, str]], *, num_frames: int, height: int, width: int):
    videos = [read_video_uniform(item["video_path"], num_frames=num_frames, height=height, width=width) for item in batch]
    prompts = [item["prompt"] for item in batch]
    paths = [item["video_path"] for item in batch]
    return {"video": torch.stack(videos, dim=0), "prompt": prompts, "video_path": paths}


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def load_neodragon_train_modules(cfg, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import DIT_ID, VAE_ID
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    dit = PyramidMMDiT.from_pretrained(f"{local_model_path}/{DIT_ID}", torch_dtype=dtype).to(device)
    vae = AsymmetricCausalVideoVAE.from_pretrained(f"{local_model_path}/{VAE_ID}", torch_dtype=dtype).to(device).eval()
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    for param in vae.parameters():
        param.requires_grad_(False)
    return dit, vae, scheduler, DEFAULT_PROMPT_MODIFIER


def load_bridge(cfg, ckpt_path: str, device: torch.device, dtype: torch.dtype) -> MobileOVNeodragonTextBridge:
    bridge = MobileOVNeodragonTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("bridge", ckpt.get("student_state", ckpt))
    bridge.load_state_dict(state, strict=False)
    for param in bridge.parameters():
        param.requires_grad_(False)
    return bridge


def scale_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    from neodragon.utils.generation_utils import (
        VAE_SCALE_FACTOR,
        VAE_SHIFT_FACTOR,
        VAE_VIDEO_SCALE_FACTOR,
        VAE_VIDEO_SHIFT_FACTOR,
    )

    latents = latents.clone()
    latents[:, :, :1] = (latents[:, :, :1] - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR
    if latents.shape[2] > 1:
        latents[:, :, 1:] = (latents[:, :, 1:] - VAE_VIDEO_SHIFT_FACTOR) * VAE_VIDEO_SCALE_FACTOR
    return latents


def freeze_dit_for_last_n_blocks(dit: torch.nn.Module, last_n: int) -> None:
    if last_n <= 0:
        return
    for param in dit.parameters():
        param.requires_grad_(False)
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("Cannot find dit.transformer_blocks for --train-last-n-blocks")
    for block in list(blocks)[-last_n:]:
        for param in block.parameters():
            param.requires_grad_(True)
    for name in ["norm_out", "proj_out"]:
        module = getattr(dit, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad_(True)


def wrap_dit(
    dit: torch.nn.Module,
    *,
    parallel: str,
    device: torch.device,
    local_rank: int,
) -> torch.nn.Module:
    parallel = parallel.lower()
    if parallel == "none":
        return dit
    if parallel == "ddp":
        return DDP(dit, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if parallel == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        strategy = ShardingStrategy.FULL_SHARD if torch.distributed.get_world_size() > 1 else ShardingStrategy.NO_SHARD
        return FSDP(
            dit,
            device_id=device,
            use_orig_params=True,
            sharding_strategy=strategy,
            sync_module_states=False,
        )
    if parallel == "deepspeed":
        return dit
    raise ValueError(f"Unknown --parallel={parallel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--manifest", required=True, help="CSV with video_path/path/mp4 and prompt/caption/text columns.")
    parser.add_argument("--bridge-ckpt", required=True)
    parser.add_argument("--output-dir", default="output/neodragon_dit_bridge_train")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--train-last-n-blocks", type=int, default=1, help="0 trains the full DiT.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp", "deepspeed"], default="none")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_USE_ORIG_PARAMS", "true")
    ctx = setup_distributed()
    if args.parallel in {"ddp", "fsdp"} and not ctx.is_distributed:
        rank0_print(ctx, f"Warning: --parallel={args.parallel} requested with WORLD_SIZE=1; falling back to --parallel=none.")
        args.parallel = "none"

    cfg = load_config(args.config)
    if args.dtype:
        cfg.backend.dtype = args.dtype
        cfg.train.dtype = args.dtype
    dtype = dtype_from_name(cfg.backend.dtype)
    if ctx.device.type == "cpu":
        dtype = torch.float32
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    dataset = VideoPromptDataset(args.manifest, max_samples=args.max_samples)
    sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True) if ctx.is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_video_batch(
            b,
            num_frames=cfg.data.frame_num,
            height=cfg.data.height,
            width=cfg.data.width,
        ),
    )
    batches = cycle_loader(loader, sampler)

    rank0_print(
        ctx,
        f"Neodragon DiT train: parallel={args.parallel} world_size={ctx.world_size} "
        f"batch_per_gpu={args.batch_size} samples={len(dataset)} dtype={dtype}",
    )
    dit, vae, scheduler, prompt_modifier = load_neodragon_train_modules(cfg, ctx.device, dtype)
    bridge = load_bridge(cfg, args.bridge_ckpt, ctx.device, dtype)
    freeze_dit_for_last_n_blocks(dit, args.train_last_n_blocks)
    dit.train()
    trainable_count = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise RuntimeError("No trainable DiT parameters.")
    rank0_print(ctx, f"Trainable DiT parameters: {trainable_count:,}")

    dit_model = wrap_dit(dit, parallel=args.parallel, device=ctx.device, local_rank=ctx.local_rank)
    deepspeed_engine = None
    opt = None
    if args.parallel == "deepspeed":
        import deepspeed

        ds_config = build_deepspeed_config(
            micro_batch_size=args.batch_size,
            world_size=ctx.world_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero_stage=args.deepspeed_zero_stage,
            dtype=dtype,
        )
        if ctx.is_main:
            write_deepspeed_config(ds_config, out_dir)
        trainable = [p for p in dit_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
        deepspeed_engine, opt, _, _ = deepspeed.initialize(
            model=dit_model,
            optimizer=opt,
            model_parameters=trainable,
            config=ds_config,
            dist_init_required=False if torch.distributed.is_initialized() else None,
        )
        dit_model = deepspeed_engine
    else:
        opt = torch.optim.AdamW((p for p in dit_model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)

    from neodragon.utils.generation_utils import _get_pyramid_latent, _prepare_past_condition_latents

    history: list[dict[str, float]] = []
    pbar = tqdm(range(1, args.steps + 1), desc="Train Neodragon DiT with Mobile-OV bridge", disable=not ctx.is_main)
    for step in pbar:
        batch = next(batches)
        video = batch["video"].to(device=ctx.device, dtype=dtype)
        prompts = [p + prompt_modifier for p in batch["prompt"]]

        with torch.no_grad():
            bridge_tokens, bridge_mask, pooled = bridge(prompts)
            encoder_hidden_states = bridge_tokens
            latents = vae.encode(video, temporal_chunk=True).latent_dist.sample()
            latents = scale_vae_latents(latents).to(dtype=dtype)

        latent_t = int(latents.shape[2])
        if latent_t < 2:
            raise RuntimeError(f"Need at least two latent frames for hybrid video training, got {latent_t}")
        unit_index = int(torch.randint(1, latent_t, (1,), device=ctx.device).item())
        stage = int(torch.randint(0, scheduler.config.stages, (1,), device=ctx.device).item())
        past_units = [latents[:, :, i : i + 1] for i in range(unit_index)]
        past_conditions = _prepare_past_condition_latents(
            past_units,
            num_stages=scheduler.config.stages,
            do_classifier_free_guidance=False,
        )
        clean_full = latents[:, :, unit_index : unit_index + 1]
        clean_stage = _get_pyramid_latent(clean_full, scheduler.config.stages)[stage].to(dtype=dtype)
        noise = torch.randn_like(clean_stage)
        t_idx = torch.randint(0, scheduler.config.num_train_timesteps, (clean_stage.shape[0],), device=ctx.device)
        sigmas = scheduler.sigmas_per_stage[stage].to(device=ctx.device, dtype=dtype)[t_idx]
        timestep = scheduler.timesteps_per_stage[stage].to(device=ctx.device, dtype=dtype)[t_idx]
        while sigmas.dim() < clean_stage.dim():
            sigmas = sigmas.view(-1, *([1] * (clean_stage.dim() - 1)))
        noisy = sigmas * noise + (1.0 - sigmas) * clean_stage
        target_flow = noise - clean_stage

        stage_input = past_conditions[stage] + [noisy]
        pred = dit_model(
            sample=[stage_input],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=bridge_mask,
            pooled_projections=pooled,
            timestep_ratio=timestep,
        )[0]
        loss = F.mse_loss(pred.float(), target_flow.float())

        if deepspeed_engine is not None:
            deepspeed_engine.backward(loss)
            deepspeed_engine.step()
        else:
            assert opt is not None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad_norm > 0:
                if args.parallel == "fsdp" and hasattr(dit_model, "clip_grad_norm_"):
                    dit_model.clip_grad_norm_(args.clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(dit_model.parameters(), args.clip_grad_norm)
            opt.step()

        if step % args.log_every == 0 or step == 1:
            item = {
                "step": float(step),
                "loss": scalar_mean(loss.detach(), ctx),
                "unit_index": float(unit_index),
                "stage": float(stage),
                "latent_t": float(latent_t),
                "trainable_params": float(trainable_count),
                "world_size": float(ctx.world_size),
            }
            if ctx.is_main:
                history.append(item)
                pbar.set_postfix(loss=f"{item['loss']:.4f}", unit=unit_index, stage=stage)

        if step % args.save_every == 0 or step == args.steps:
            state = full_state_dict(dit_model, args.parallel)
            if ctx.is_main:
                torch.save(
                    {
                        "step": step,
                        "dit": state,
                        "bridge_ckpt": args.bridge_ckpt,
                        "config": cfg,
                        "args": vars(args),
                        "history": history,
                        "parallel": {
                            "backend": args.parallel,
                            "world_size": ctx.world_size,
                            "deepspeed_zero_stage": args.deepspeed_zero_stage if args.parallel == "deepspeed" else None,
                        },
                    },
                    out_dir / "neodragon_dit_bridge_latest.pt",
                )
                (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            barrier()

    rank0_print(ctx, f"Saved DiT checkpoint to {out_dir / 'neodragon_dit_bridge_latest.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
