#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "sana"))

from diffusion.model.wan.vae import WanVAE

from new_mobile_ov.bridge import MobileOVTextBridge
from new_mobile_ov.config import load_config
from new_mobile_ov.generation.backends.mobile_o_sana_0_5b import MobileOSana05BBackend
from new_mobile_ov.motion import LatentMotionWeaver
from new_mobile_ov.training.losses import latent_motion_weaver_loss


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def center_crop_resize_frames(frames: list[np.ndarray], target_size: tuple[int, int]) -> torch.Tensor:
    h, w = frames[0].shape[:2]
    target_h, target_w = target_size
    ratio = float(target_w) / float(target_h)
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))
    transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return torch.stack([transform(Image.fromarray(frame)) for frame in frames], dim=0)


def read_video_frames(path: Path, frame_num: int, target_size: tuple[int, int], sampling_rate: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rate = max(1, int(sampling_rate))
    while total < frame_num * rate and rate > 1:
        rate -= 1
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= frame_num:
                break
        idx += 1
    cap.release()
    if len(frames) != frame_num:
        raise RuntimeError(f"Need {frame_num} frames, got {len(frames)} from {path}")
    return center_crop_resize_frames(frames, target_size)


def image_to_static_frames(image: Image.Image, frame_num: int, target_size: tuple[int, int]) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"))
    return center_crop_resize_frames([arr] * frame_num, target_size)


def save_video(frames_cthw: torch.Tensor, path: Path, fps: int = 8) -> None:
    frames = ((frames_cthw.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
    frames = frames.permute(1, 2, 3, 0).numpy()  # T,H,W,C
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def build_lmw(cfg, device: torch.device, dtype: torch.dtype, hidden_dim: int | None, blocks: int | None) -> LatentMotionWeaver:
    model = LatentMotionWeaver(
        latent_channels=cfg.motion.latent_channels,
        text_dim=cfg.motion.text_dim,
        hidden_dim=hidden_dim or cfg.motion.hidden_dim,
        temporal_len=cfg.motion.temporal_len,
        num_blocks=blocks or cfg.motion.num_blocks,
        mlp_ratio=cfg.motion.mlp_ratio,
        temporal_kernel=cfg.motion.temporal_kernel,
        temporal_dilation_cycle=cfg.motion.temporal_dilation_cycle,
    )
    return model.to(device=device, dtype=dtype)


def decode_latent(vae: WanVAE, z: torch.Tensor) -> torch.Tensor:
    return vae.decode([z.detach().float().to(vae.device)])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-config", default="configs/mobile_ov_current.yaml")
    parser.add_argument("--mobileo-config", default="configs/mobile_o_sana_0_5b.yaml")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--vae-path", default="/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth")
    parser.add_argument("--prompt", default="A person moving in a short video.")
    parser.add_argument("--output-dir", default="output/infer_two_options_lmw_smoke")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-rate", type=int, default=3)
    parser.add_argument("--quick-train-steps", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--skip-mobileo", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {"device": str(device), "dtype": str(dtype)}

    cfg = load_config(args.current_config)
    mobileo_cfg = load_config(args.mobileo_config)
    cfg.motion.hidden_dim = int(args.hidden_dim)
    cfg.motion.num_blocks = int(args.blocks)
    mobileo_cfg.motion.hidden_dim = int(args.hidden_dim)
    mobileo_cfg.motion.num_blocks = int(args.blocks)

    print("Loading WanVAE...")
    vae = WanVAE(vae_pth=args.vae_path, device=device)
    print("Encoding reference video...")
    ref_frames = read_video_frames(Path(args.video_path), args.frame_num, (args.height, args.width), args.sampling_rate)
    with torch.no_grad():
        z_gt = vae.encode([ref_frames.transpose(0, 1).to(device)])[0].unsqueeze(0).to(device=device, dtype=dtype)
    metrics["z_gt_shape"] = list(z_gt.shape)

    print("Loading Mobile-OV bridge...")
    bridge = MobileOVTextBridge(cfg.bridge, device=device, dtype=dtype).eval()
    with torch.no_grad():
        prompt_embeds, prompt_mask, text = bridge.encode([args.prompt])
    metrics["prompt_embeds_shape"] = list(prompt_embeds.shape)
    metrics["text_shape"] = list(text.shape)

    print("Building and quick-training LMW...")
    lmw = build_lmw(cfg, device, dtype, args.hidden_dim, args.blocks)
    opt = torch.optim.AdamW(lmw.parameters(), lr=1e-4)
    train_losses = []
    for _ in trange(args.quick_train_steps, desc="quick train LMW"):
        z_pred = lmw(z_gt[:, :, 0], text)
        loss, loss_metrics = latent_motion_weaver_loss(z_pred.float(), z_gt.float(), motion_weight=0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.detach().cpu()))
    metrics["quick_train_losses"] = train_losses

    print("Option 1: current Mobile-OV text + GT WanVAE anchor + LMW")
    with torch.no_grad():
        z_pred_current = lmw(z_gt[:, :, 0], text)[0]
        z_copy = z_gt[0, :, 0:1].repeat(1, z_gt.shape[2], 1, 1)
        current_video = decode_latent(vae, z_pred_current)
        copy_video = decode_latent(vae, z_copy)
        gt_video = decode_latent(vae, z_gt[0].float())
    save_video(current_video, out_dir / "option1_mobile_ov_current_lmw_pred.mp4")
    save_video(copy_video, out_dir / "baseline_copy_anchor.mp4")
    save_video(gt_video, out_dir / "reference_gt_decode.mp4")
    metrics["option1_video"] = str(out_dir / "option1_mobile_ov_current_lmw_pred.mp4")

    if not args.skip_mobileo:
        print("Option 2: Mobile-O 0.5B native image anchor + WanVAE static anchor + LMW")
        t0 = time.time()
        mobileo = MobileOSana05BBackend(
            model_path=mobileo_cfg.backend.model_path,
            device=device,
            dtype=dtype,
        )
        image = mobileo.generate_image_from_prompt(args.prompt)
        image_path = out_dir / "option2_mobile_o_anchor.png"
        image.save(image_path)
        static_frames = image_to_static_frames(image, args.frame_num, (args.height, args.width))
        with torch.no_grad():
            z_static = vae.encode([static_frames.transpose(0, 1).to(device)])[0].unsqueeze(0).to(device=device, dtype=dtype)
            z_pred_mobileo = lmw(z_static[:, :, 0], text)[0]
            mobileo_video = decode_latent(vae, z_pred_mobileo)
            mobileo_copy_video = decode_latent(vae, z_static[0, :, 0:1].repeat(1, z_static.shape[2], 1, 1))
        save_video(mobileo_video, out_dir / "option2_mobile_o_sana_0_5b_lmw_pred.mp4")
        save_video(mobileo_copy_video, out_dir / "option2_mobile_o_static_copy.mp4")
        metrics["option2_anchor_image"] = str(image_path)
        metrics["option2_video"] = str(out_dir / "option2_mobile_o_sana_0_5b_lmw_pred.mp4")
        metrics["option2_seconds"] = time.time() - t0

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
