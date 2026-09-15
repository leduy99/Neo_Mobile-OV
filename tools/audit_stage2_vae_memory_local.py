#!/usr/bin/env python
"""Compare native 16/8-frame causal VAE chunks on all training aspect buckets."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from new_mobile_ov.config import load_config
from new_mobile_ov.training.stage1_alignment import JsonlRecords, prepare_sample
from new_mobile_ov.training.stage1_alignment_data import load_sources
from new_mobile_ov.training.stage1_vae import clear_temporal_cache, encode_posterior
from tools.data_prepare.download_alignment_images import atomic_json
from tools.train_neodragon_dit_bridge import load_neodragon_train_modules


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("output/alignment_data_checks/stage1_slurm_release"))
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the VAE memory audit inside SLURM")
    device = torch.device("cuda", 0)
    dit, vae, _, _ = load_neodragon_train_modules(load_config(args.config), device, torch.bfloat16,
                                                target_stack="multistep")
    del dit
    gc.collect()
    torch.cuda.empty_cache()
    sources = load_sources(args.data_root)
    record = JsonlRecords(args.data_root / "validation" / "t2v.jsonl")[0]
    sample = prepare_sample(record, sources, 320, 512)
    frames = sample["video"].permute(1, 0, 2, 3)
    rows = []
    for height, width in ((320, 512), (512, 320), (384, 384)):
        video = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device, torch.bfloat16)
        versions = {}
        measurements = {}
        for window in (16, 8):
            clear_temporal_cache(vae)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if window == 16:
                    posterior = vae.encode(video, temporal_chunk=True).latent_dist
                else:
                    posterior = encode_posterior(vae, video, window_size=8)
            versions[window] = posterior.parameters.float().cpu()
            assert posterior.mean.shape[2] == 7
            measurements[str(window)] = dict(
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                post_encode_allocated_gib=torch.cuda.memory_allocated(device) / 1024**3)
            del posterior
        reference, candidate = versions[16], versions[8]
        rms = (candidate - reference).square().mean().sqrt()
        relative_rms = float(rms / reference.square().mean().sqrt().clamp_min(1e-8))
        cosine = float(F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0))
        # Check each temporal unit, not just a whole-clip average that could hide seam errors.
        unit_relative_rms = [float((candidate[:, :, i] - reference[:, :, i]).square().mean().sqrt()
                                  / reference[:, :, i].square().mean().sqrt().clamp_min(1e-8)) for i in range(7)]
        assert torch.isfinite(candidate).all()
        assert max(unit_relative_rms) < 0.005, unit_relative_rms
        assert cosine > 0.9999, cosine
        assert measurements["8"]["peak_allocated_gib"] < measurements["16"]["peak_allocated_gib"]
        assert all(not len(getattr(module, "cache_front_feat", ())) for module in vae.modules())
        row = dict(height=height, width=width, frames=49, latent_units=7,
                   posterior_relative_rms=relative_rms, posterior_cosine=cosine,
                   unit_relative_rms=unit_relative_rms, memory=measurements)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del video
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "vae_memory_audit.json", dict(passed=True, sample_id=record["sample_id"],
                note="Numerical/memory audit on one clip, three shapes; not a generation-quality evaluation", rows=rows))


if __name__ == "__main__":
    main()
