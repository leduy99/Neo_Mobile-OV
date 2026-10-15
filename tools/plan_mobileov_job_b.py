#!/usr/bin/env python
"""Check the same generation parent/release as Job A plus a prepared QA release."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plan_mobileov_stage2_first_frame import plan, PARENT_SHA256
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file
from new_mobile_ov.training.job_b_qa import verify_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integration-smoke", action="store_true")
    args = parser.parse_args(argv)
    args.parent_sha256, args.parent_step = PARENT_SHA256, 283977
    args.world_size, args.accumulation, args.seed = 8, 2, 20260911
    args.min_train_videos, args.require_expanded_release, args.audit_samples = 500000, True, 0
    destination, args.output = args.output, None
    report = plan(args)
    replay = verify_release(args.qa_root)
    report.update(format="mobileov_job_b_plan_v1", lora=True, kl=False,
                  lora_rank=16, lora_alpha=32, lora_lr=1e-5, lora_warmup=1000,
                  qa_weight=.1, qa_every=8, qa_ratio=[.5, .2, .3],
                  qa_presentations=(report["additional_updates"] // 8) * 8,
                  qa_summary_sha256=sha256_file(args.qa_root / "qa_summary.json"),
                  qa_counts={name: value["rows"] for name, value in replay["files"].items()},
                  qa_validation="heldout_answer_CE; not_understanding_accuracy_or_forgetting_guarantee")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()) != report:
        raise ValueError("Job-B plan changed; preserve the original run")
    atomic_json(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
