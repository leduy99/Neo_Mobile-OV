#!/usr/bin/env python
"""Fail closed before torchrun unless the original Stage-2 Adam state is present."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from new_mobile_ov.training.stage2_epoch_extension import optimizer_audit, validate_extension
from new_mobile_ov.training.stage2_media_epochs import plan_from_release
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file


def run(args):
    if not args.resume.is_file():
        raise FileNotFoundError(f"Missing {args.resume}; no weights-only/HF fallback. Restore the original stage2_resume.pt.")
    before = args.resume.stat()
    saved = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
    world_size, accumulation = getattr(args, "world_size", 8), getattr(args, "accumulation", 2)
    repartition = getattr(args, "repartition_media_batch", False)
    plan = plan_from_release(args.data_root, world_size=world_size, accumulation=accumulation, epochs=args.epochs,
                             seed=20260911, min_videos=500000, require_expansion=True,
                             allow_partial_cycle=True, rng_world_size=getattr(args, "media_rng_world_size", None))
    source = saved.get("contract", {})
    if (saved.get("format") != source.get("format") or not saved.get("connector") or not saved.get("dit")
            or source.get("data_summary_sha256") != sha256_file(args.data_root / "stage1_summary.json")):
        raise ValueError("Resume weights, format or original generation release are missing/mismatched")
    target = dict(source, steps=plan.steps, media_epochs=plan.contract(),
                  world_size=world_size, accumulation=accumulation)
    validate_extension(source, target, saved.get("step"), repartition=repartition)
    audit = optimizer_audit(saved.get("optimizer", {}), saved["step"], source["lr"])
    if source["media_epochs"]["epochs"] == 3:
        if args.output_dir.resolve() == args.resume.resolve().parent:
            raise ValueError("Use a new output directory; preserve the original three-epoch run")
    elif not isinstance(saved.get("rng_states"), list) or len(saved["rng_states"]) != world_size:
        raise ValueError("Extension resume must contain per-rank RNG states")
    current = args.output_dir / "run_contract.json"
    if current.exists() and json.loads(current.read_text()) != target:
        raise ValueError("Existing output belongs to a different run/target; use its own resume or a new directory")
    if current.exists() and args.resume.resolve() != (args.output_dir / "stage2_resume.pt").resolve():
        raise ValueError("Existing extension output must resume its OWN stage2_resume.pt; never rewind it to epoch 3")
    after = args.resume.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Resume checkpoint changed during preflight")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget = dict(plan.contract(), data_summary_sha256=source["data_summary_sha256"])
    path = args.output_dir / "epoch_plan.json"
    if path.exists() and json.loads(path.read_text()) != budget:
        raise ValueError("Existing extension budget differs; preserve its output")
    atomic_json(path, budget)
    receipt = dict(resume=str(args.resume.resolve()), resume_step=saved["step"], target_step=plan.steps,
                   total_media_epochs=args.epochs, additional_updates=plan.steps - saved["step"],
                   global_rng_available="rng_states" in saved,
                   ddp_repartitioned=source["world_size"] != world_size,
                   source_world_size=source["world_size"], world_size=world_size,
                   sampler_next_micro=saved["step"] * accumulation,
                   epoch_checkpoints={str(epoch): epoch * plan.steps_per_epoch for epoch in range(4, args.epochs + 1)},
                   **audit)
    atomic_json(args.output_dir / f"resume_preflight_step{saved['step']:06d}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(4, 5), default=5)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--media-rng-world-size", type=int)
    parser.add_argument("--repartition-media-batch", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inspect this large optimizer checkpoint inside the submitted GPU job, not on a login node")
    return run(args)


if __name__ == "__main__":
    main()
