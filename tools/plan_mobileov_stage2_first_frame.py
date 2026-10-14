#!/usr/bin/env python
"""Verify a completed epoch-3 parent and budget its one-pass first-frame fork."""
import argparse
import json
from pathlib import Path
import random
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_mobile_ov.training.stage2_media_epochs import plan_from_release, validate_fork_parent, first_frame_image
from new_mobile_ov.training.stage1_alignment import JsonlRecords, prepare_sample
from new_mobile_ov.training.stage1_alignment_data import load_sources
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file

PARENT_SHA256 = "a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053"


def write_data_audit(root, output, *, samples=8, seed=20260911):
    """Representative processed media, not an automated semantic-verification claim."""
    output.mkdir(parents=True, exist_ok=True)
    sources = load_sources(root)
    rows = []
    for task in ("t2i", "t2v"):
        records = JsonlRecords(root / "train" / f"{task}.jsonl")
        selected = random.Random(seed + (1 if task == "t2v" else 0)).sample(range(len(records)), min(samples, len(records)))
        for ordinal, index in enumerate(selected):
            sample = prepare_sample(records[index], sources)
            video = sample["video"]
            positions = [0] if task == "t2i" else [0, video.shape[1] // 2, video.shape[1] - 1]
            width, height = video.shape[-1], video.shape[-2]
            preview = Image.new("RGB", (width * len(positions), height + 24), "white")
            draw = ImageDraw.Draw(preview)
            for column, position in enumerate(positions):
                preview.paste(first_frame_image(video[:, position:position + 1]), (column * width, 24))
                draw.text((column * width + 4, 4), f"sampled frame {position}", fill="black")
            filename = f"{task}_{ordinal:02d}.png"
            preview.save(output / filename)
            rows.append(dict(task=task, sample_id=sample["sample_id"], prompt=sample["prompt"], preview=filename,
                             source_frame_indices=[sample["frame_indices"][p] for p in positions],
                             source_duration_seconds=sample.get("source_duration_seconds")))
    report = dict(semantics_verified=False, policy="inspect caption vs first/middle/last; no automatic filtering/relabeling",
                  data_summary_sha256=sha256_file(root / "stage1_summary.json"), seed=seed, samples=rows)
    atomic_json(output / "audit.json", report)
    lines = ["# Processed Training Samples", "", "Decode checks do NOT verify caption semantics.", ""]
    for row in rows:
        lines.extend([f"## {row['sample_id']}", row["prompt"], "", f"![processed sample]({row['preview']})", ""])
    (output / "audit.md").write_text("\n".join(lines))
    return report


def plan(args):
    digest = sha256_file(args.parent)
    if args.parent_sha256 and digest != args.parent_sha256:
        raise ValueError("Parent SHA256 differs; restore the selected final epoch-3 checkpoint")
    payload = torch.load(args.parent, map_location="cpu", weights_only=False, mmap=True)
    epochs = plan_from_release(args.data_root, world_size=args.world_size, accumulation=args.accumulation,
                               epochs=1, start_epoch=3, seed=args.seed, min_videos=args.min_train_videos,
                               require_expansion=args.require_expanded_release)
    summary_hash = sha256_file(args.data_root / "stage1_summary.json")
    validate_fork_parent(payload, epochs, expected_step=args.parent_step, data_summary_sha256=summary_hash,
                         integration_smoke=args.integration_smoke)
    report = dict(format="stage2_first_frame_focus_plan_v1", parent_sha256=digest, parent_step=args.parent_step,
                  data_summary_sha256=summary_hash, media_epochs=epochs.contract(),
                  additional_updates=epochs.steps, absolute_final_step=args.parent_step + epochs.steps,
                  loss_weights=dict(t2i=.5, video=.5, reconstruction=.2), t2v_first_probability=.5,
                  history_noise=0, ema=False, lora=False, teacher=False, optimizer="fresh_AdamW_local_warmup",
                  integration_smoke=args.integration_smoke, caption_semantics_verified=False)
    if args.output:
        if args.output.exists() and json.loads(args.output.read_text()) != report:
            raise ValueError("Existing fork plan differs; use a new output directory")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        if args.audit_samples:
            write_data_audit(args.data_root, args.output.parent / "data_audit", samples=args.audit_samples, seed=args.seed)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-step", type=int, default=283977)
    parser.add_argument("--parent-sha256", default=PARENT_SHA256)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--min-train-videos", type=int, default=500000)
    parser.add_argument("--require-expanded-release", action="store_true")
    parser.add_argument("--integration-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-samples", type=int, default=8, help="Per source; 0 skips representative decode previews")
    args = parser.parse_args(argv)
    if not 0 <= args.audit_samples <= 100 or args.parent_step < 1:
        parser.error("Invalid audit count or parent step")
    print(json.dumps(plan(args), indent=2))


if __name__ == "__main__":
    main()
