#!/usr/bin/env python
"""CPU-only, verified media counts and exact optimizer budget; no model loading."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_mobile_ov.training.stage2_media_epochs import plan_from_release
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--min-train-videos", type=int, default=500000)
    parser.add_argument("--require-expanded-release", action="store_true")
    parser.add_argument("--output", type=Path, help="Persist the verified budget; refuse to change an existing plan")
    args = parser.parse_args()
    plan = plan_from_release(args.data_root, world_size=args.world_size, accumulation=args.accumulation,
                             epochs=args.epochs, seed=args.seed, min_videos=args.min_train_videos,
                             require_expansion=args.require_expanded_release)
    payload = dict(plan.contract(), data_summary_sha256=sha256_file(args.data_root / "stage1_summary.json"))
    if args.output:
        if args.output.exists() and json.loads(args.output.read_text()) != payload:
            raise ValueError("Existing training budget differs; use a new run directory")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
