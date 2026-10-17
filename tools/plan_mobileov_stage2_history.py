#!/usr/bin/env python
"""Pin the epoch-3 parent and unchanged media budget for uniform-history training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import plan_mobileov_stage2_first_frame as parent_plan
from tools.data_prepare.download_alignment_images import atomic_json
from new_mobile_ov.training.stage2_history import MAX_HISTORY_SIGMA, history_contract


def plan(args):
    destination, args.output = args.output, None
    try:
        result = parent_plan.plan(args)
    finally:
        args.output = destination
    result.update(format="stage2_uniform_history_plan_v1", t2v_first_probability=None,
                  loss_weights=dict(t2i=.5, video=.5, reconstruction=.1),
                  history_noise=MAX_HISTORY_SIGMA, history_corruption=history_contract(MAX_HISTORY_SIGMA))
    if destination:
        if destination.exists() and json.loads(destination.read_text()) != result:
            raise ValueError("Existing uniform-history plan differs; do not change a running experiment")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-step", type=int, default=283977)
    parser.add_argument("--parent-sha256", default=parent_plan.PARENT_SHA256)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--min-train-videos", type=int, default=500000)
    parser.add_argument("--require-expanded-release", action="store_true")
    parser.add_argument("--integration-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.audit_samples = 0
    print(json.dumps(plan(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
