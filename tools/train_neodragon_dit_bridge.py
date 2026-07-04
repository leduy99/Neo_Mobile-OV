#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
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
    def __init__(
        self,
        manifest: str | Path,
        max_samples: int = -1,
        *,
        caption_aug: bool = False,
        caption_variant_columns: list[str] | None = None,
        caption_variant_weights: list[float] | None = None,
        caption_fallback_column: str = "caption",
    ):
        self.manifest_path = Path(manifest)
        header = pd.read_csv(manifest, nrows=0)
        video_col = next((c for c in ["video_path", "path", "mp4"] if c in header.columns), None)
        latent_col = "latent_path" if "latent_path" in header.columns else None
        prompt_col = next((c for c in ["prompt", "caption", "text"] if c in header.columns), None)
        if video_col is None and latent_col is None:
            raise ValueError(f"{manifest} must contain latent_path or one of columns: video_path, path, mp4")
        if prompt_col is None:
            raise ValueError(f"{manifest} must contain one of columns: prompt, caption, text")
        keep_cols = [
            video_col or "",
            latent_col or "",
            prompt_col,
            "caption",
            "caption_short",
            "caption_medium",
            "caption_long",
            "clip_start_sec",
            "clip_end_sec",
            "clip_num_frames",
            "clip_fps",
        ]
        keep_cols = [c for c in dict.fromkeys(keep_cols) if c in header.columns]
        df = pd.read_csv(manifest, usecols=keep_cols)
        required_col = latent_col or video_col
        if required_col is not None:
            df = df.dropna(subset=[required_col])
        if max_samples > 0:
            df = df.head(max_samples)
        self.rows = df.to_dict("records")
        self.video_col = video_col
        self.latent_col = latent_col
        self.prompt_col = prompt_col
        self.has_latents = latent_col is not None
        self.caption_aug = bool(caption_aug)
        self.caption_variant_columns = caption_variant_columns or ["caption_short", "caption_medium", "caption_long"]
        self.caption_variant_weights = caption_variant_weights
        self.caption_fallback_column = caption_fallback_column
        if not self.rows:
            raise ValueError(f"No samples found in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _valid_text(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value).strip()

    def _choose_prompt(self, row: dict[str, object]) -> str:
        if self.caption_aug:
            variants: list[str] = []
            weights: list[float] = []
            for idx, col in enumerate(self.caption_variant_columns):
                text = self._valid_text(row.get(col))
                if not text:
                    continue
                variants.append(text)
                if self.caption_variant_weights and idx < len(self.caption_variant_weights):
                    weights.append(float(self.caption_variant_weights[idx]))
                else:
                    weights.append(1.0)
            if variants:
                return random.choices(variants, weights=weights, k=1)[0]
        return (
            self._valid_text(row.get(self.prompt_col))
            or self._valid_text(row.get(self.caption_fallback_column))
            or self._valid_text(row.get("caption"))
        )

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        item = {
            "video_path": str(row.get(self.video_col) or "") if self.video_col else "",
            "prompt": self._choose_prompt(row),
            "clip_start_sec": row.get("clip_start_sec", 0.0),
            "clip_end_sec": row.get("clip_end_sec", 0.0),
            "clip_num_frames": row.get("clip_num_frames", 0),
            "clip_fps": row.get("clip_fps", 0.0),
        }
        if self.latent_col:
            item["latent_path"] = str(row[self.latent_col])
        return item


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def center_crop_resize_rgb(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target_ar = float(width) / float(height)
    src_ar = float(w) / float(h)
    if src_ar > target_ar:
        new_w = int(round(h * target_ar))
        x0 = max((w - new_w) // 2, 0)
        frame = frame[:, x0 : x0 + new_w]
    elif src_ar < target_ar:
        new_h = int(round(w / target_ar))
        y0 = max((h - new_h) // 2, 0)
        frame = frame[y0 : y0 + new_h, :]
    image = Image.fromarray(frame).convert("RGB").resize((width, height), Image.BICUBIC)
    return np.asarray(image)


def read_video_clip(
    path: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    target_fps: float,
    clip_start_sec: float = 0.0,
) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_sec = max(float(clip_start_sec), 0.0)
    times = start_sec + np.arange(num_frames, dtype=np.float64) / max(float(target_fps), 1e-6)
    indices = np.rint(times * src_fps).astype(np.int64)
    if total_frames > 0:
        indices = np.clip(indices, 0, total_frames - 1)

    out = []
    last_rgb = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if last_rgb is None:
                continue
            rgb = last_rgb
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_rgb = rgb
        rgb = center_crop_resize_rgb(rgb, height=height, width=width)
        arr = rgb.astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    cap.release()
    if not out:
        raise RuntimeError(f"Could not read frames from {path}")
    while len(out) < num_frames:
        out.append(out[-1].clone())
    return torch.stack(out[:num_frames], dim=1)  # [C,T,H,W]


def load_latent_tensor(path_value: str, *, latent_root: Path) -> torch.Tensor:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = latent_root / path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    latent = payload
    if isinstance(payload, dict):
        latent = payload.get("latent", payload.get("latents", payload.get("z", None)))
    if not isinstance(latent, torch.Tensor):
        raise RuntimeError(f"Could not find latent tensor in {path}")
    if latent.dim() == 5:
        if latent.shape[0] != 1:
            raise RuntimeError(f"Expected single-sample latent in {path}, got {tuple(latent.shape)}")
        latent = latent[0]
    if latent.dim() != 4:
        raise RuntimeError(f"Expected latent [C,T,H,W] in {path}, got {tuple(latent.shape)}")
    return latent.contiguous()


def collate_video_batch(
    batch: list[dict[str, str]],
    *,
    num_frames: int,
    height: int,
    width: int,
    target_fps: float,
    latent_root: Path | None = None,
    use_latents: bool = False,
):
    prompts = [item["prompt"] for item in batch]
    paths = [item.get("video_path", "") for item in batch]
    if use_latents:
        if latent_root is None:
            raise ValueError("latent_root is required when use_latents=True")
        latents = [load_latent_tensor(item["latent_path"], latent_root=latent_root) for item in batch]
        return {
            "latents": torch.stack(latents, dim=0),
            "prompt": prompts,
            "video_path": paths,
            "latent_path": [item["latent_path"] for item in batch],
        }

    videos = []
    for item in batch:
        item_num_frames = _safe_int(item.get("clip_num_frames"), num_frames) or num_frames
        item_fps = _safe_float(item.get("clip_fps"), target_fps) or target_fps
        videos.append(
            read_video_clip(
                item["video_path"],
                num_frames=item_num_frames,
                height=height,
                width=width,
                target_fps=item_fps,
                clip_start_sec=_safe_float(item.get("clip_start_sec"), 0.0),
            )
        )
    return {"video": torch.stack(videos, dim=0), "prompt": prompts, "video_path": paths}


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def load_neodragon_train_modules(cfg, device: torch.device, dtype: torch.dtype, *, load_vae: bool = True):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import DIT_ID
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    dit = PyramidMMDiT.from_pretrained(f"{local_model_path}/{DIT_ID}", torch_dtype=dtype).to(device)
    scheduler = PyramidFlowMatchEulerDiscreteScheduler()
    vae = None
    if load_vae:
        from neodragon import VAE_ID
        from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

        vae = AsymmetricCausalVideoVAE.from_pretrained(f"{local_model_path}/{VAE_ID}", torch_dtype=dtype).to(device).eval()
        for param in vae.parameters():
            param.requires_grad_(False)
    return dit, vae, scheduler, DEFAULT_PROMPT_MODIFIER


def load_neodragon_teacher_modules(cfg, device: torch.device, dtype: torch.dtype):
    repo_path, _, local_model_path = ensure_neodragon_assets(
        repo_path=cfg.backend.extra.get("repo_path"),
        cache_dir=cfg.backend.extra.get("cache_dir"),
        model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
        repo_url=cfg.backend.extra.get("repo_url"),
    )
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from neodragon import CONTEXT_ADAPTER_ID, DIT_ID
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.text_encoder_bundle import TextEncoderBundle

    text_bundle = TextEncoderBundle.from_pretrained(local_model_path, torch_dtype=dtype).to(device).eval()
    context_adapter = ContextAdapter.from_pretrained(
        f"{local_model_path}/{CONTEXT_ADAPTER_ID}",
        torch_dtype=dtype,
    ).to(device).eval()
    teacher_dit = PyramidMMDiT.from_pretrained(f"{local_model_path}/{DIT_ID}", torch_dtype=dtype).to(device).eval()
    for module in [text_bundle, context_adapter, teacher_dit]:
        for param in module.parameters():
            param.requires_grad_(False)
    return text_bundle, context_adapter, teacher_dit


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
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument(
        "--use-precomputed-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-detects from latent_path column when omitted.",
    )
    parser.add_argument("--caption-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--caption-variant-columns", default="caption_short,caption_medium,caption_long")
    parser.add_argument("--caption-variant-weights", default="")
    parser.add_argument("--caption-fallback-column", default="caption")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-cos-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp", "deepspeed"], default="none")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--keep-step-checkpoints",
        action="store_true",
        help="Also keep neodragon_dit_bridge_stepXXXXXX.pt at each save interval.",
    )
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

    caption_columns = [x.strip() for x in args.caption_variant_columns.split(",") if x.strip()]
    caption_weights = [float(x.strip()) for x in args.caption_variant_weights.split(",") if x.strip()]
    if caption_weights and len(caption_weights) != len(caption_columns):
        raise ValueError("--caption-variant-weights must match --caption-variant-columns length")
    dataset = VideoPromptDataset(
        args.manifest,
        max_samples=args.max_samples,
        caption_aug=args.caption_aug,
        caption_variant_columns=caption_columns,
        caption_variant_weights=caption_weights or None,
        caption_fallback_column=args.caption_fallback_column,
    )
    use_precomputed_latents = dataset.has_latents if args.use_precomputed_latents is None else bool(args.use_precomputed_latents)
    if use_precomputed_latents and not dataset.has_latents:
        raise ValueError("--use-precomputed-latents requires a manifest with latent_path column.")
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
            target_fps=args.target_fps,
            latent_root=Path(args.manifest).expanduser().parent,
            use_latents=use_precomputed_latents,
        ),
    )
    batches = cycle_loader(loader, sampler)

    rank0_print(
        ctx,
        f"Neodragon DiT train: parallel={args.parallel} world_size={ctx.world_size} "
        f"batch_per_gpu={args.batch_size} samples={len(dataset)} dtype={dtype} "
        f"data_mode={'precomputed_latents' if use_precomputed_latents else 'online_vae'}",
    )
    dit, vae, scheduler, prompt_modifier = load_neodragon_train_modules(
        cfg,
        ctx.device,
        dtype,
        load_vae=not use_precomputed_latents,
    )
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

    teacher_text = None
    teacher_context_adapter = None
    teacher_dit = None
    if args.distill_weight > 0.0 or args.distill_cos_weight > 0.0:
        rank0_print(
            ctx,
            f"Loading frozen Neodragon teacher for functional distill: "
            f"mse_weight={args.distill_weight} cos_weight={args.distill_cos_weight}",
        )
        teacher_text, teacher_context_adapter, teacher_dit = load_neodragon_teacher_modules(cfg, ctx.device, dtype)

    from neodragon.utils.generation_utils import _get_pyramid_latent, _prepare_past_condition_latents

    history: list[dict[str, float]] = []
    pbar = tqdm(range(1, args.steps + 1), desc="Train Neodragon DiT with Mobile-OV bridge", disable=not ctx.is_main)
    for step in pbar:
        batch = next(batches)
        prompts = [p + prompt_modifier for p in batch["prompt"]]

        with torch.no_grad():
            bridge_tokens, bridge_mask, pooled = bridge(prompts)
            encoder_hidden_states = bridge_tokens
            if use_precomputed_latents:
                latents = batch["latents"].to(device=ctx.device, dtype=dtype, non_blocking=True)
            else:
                if vae is None:
                    raise RuntimeError("VAE is not loaded; cannot run online encoding.")
                video = batch["video"].to(device=ctx.device, dtype=dtype)
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
        diff_loss = F.mse_loss(pred.float(), target_flow.float())
        distill_loss = pred.new_zeros(())
        distill_cos_loss = pred.new_zeros(())
        if teacher_dit is not None and teacher_text is not None and teacher_context_adapter is not None:
            with torch.no_grad():
                teacher_tokens, teacher_mask, teacher_pooled = teacher_text(prompts, ctx.device)
                teacher_tokens = teacher_context_adapter(teacher_tokens)
                teacher_pred = teacher_dit(
                    sample=[stage_input],
                    encoder_hidden_states=teacher_tokens,
                    encoder_attention_mask=teacher_mask,
                    pooled_projections=teacher_pooled,
                    timestep_ratio=timestep,
                )[0]
            if args.distill_weight > 0.0:
                distill_loss = F.mse_loss(pred.float(), teacher_pred.float())
            if args.distill_cos_weight > 0.0:
                distill_cos_loss = 1.0 - F.cosine_similarity(
                    pred.float().reshape(pred.shape[0], -1),
                    teacher_pred.float().reshape(teacher_pred.shape[0], -1),
                    dim=-1,
                ).mean()
        loss = diff_loss + args.distill_weight * distill_loss + args.distill_cos_weight * distill_cos_loss

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
                    "diff_loss": scalar_mean(diff_loss.detach(), ctx),
                    "distill_loss": scalar_mean(distill_loss.detach(), ctx),
                    "distill_cos_loss": scalar_mean(distill_cos_loss.detach(), ctx),
                    "unit_index": float(unit_index),
                    "stage": float(stage),
                    "latent_t": float(latent_t),
                    "trainable_params": float(trainable_count),
                    "world_size": float(ctx.world_size),
            }
            if ctx.is_main:
                history.append(item)
                pbar.set_postfix(
                    loss=f"{item['loss']:.4f}",
                    diff=f"{item['diff_loss']:.4f}",
                    dist=f"{item['distill_loss']:.4f}",
                    unit=unit_index,
                    stage=stage,
                )

        if step % args.save_every == 0 or step == args.steps:
            state = full_state_dict(dit_model, args.parallel)
            if ctx.is_main:
                payload = {
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
                }
                torch.save(payload, out_dir / "neodragon_dit_bridge_latest.pt")
                if args.keep_step_checkpoints:
                    torch.save(payload, out_dir / f"neodragon_dit_bridge_step{step:06d}.pt")
                (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            barrier()

    rank0_print(ctx, f"Saved DiT checkpoint to {out_dir / 'neodragon_dit_bridge_latest.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
