from __future__ import annotations

import copy

import pytest

from new_mobile_ov.training.exp5_schedule import Exp5Schedule
from tools.train_neodragon_exp5 import build_parser, validate_resume_compatibility


def _args():
    return build_parser().parse_args(["--bridge-ckpt", "exp1.pt"])


def _schedule(args) -> Exp5Schedule:
    return Exp5Schedule(
        total_steps=args.steps,
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        dit_warmup_steps=args.dit_warmup_steps,
        final_cooldown_steps=args.final_cooldown_steps,
        bridge_warmup_steps=args.bridge_warmup_steps,
        bridge_cooldown_steps=args.bridge_cooldown_steps,
        flow_start_weight=args.flow_start_weight,
        flow_peak_weight=args.flow_weight,
        flow_final_weight=args.flow_final_weight,
    )


def _payload(args, schedule: Exp5Schedule) -> dict:
    return {
        "args": copy.deepcopy(vars(args)),
        "schedule": schedule.as_dict(),
        "parallel": {"backend": args.parallel, "world_size": 8},
    }


def test_resume_accepts_same_training_configuration() -> None:
    args = _args()
    schedule = _schedule(args)

    validate_resume_compatibility(_payload(args, schedule), args, schedule, world_size=8)


def test_resume_allows_smoke_only_run_limit_to_change() -> None:
    args = _args()
    schedule = _schedule(args)
    payload = _payload(args, schedule)
    payload["args"]["max_run_steps"] = 4

    validate_resume_compatibility(payload, args, schedule, world_size=8)


def test_resume_rejects_changed_loss_weight() -> None:
    args = _args()
    schedule = _schedule(args)
    payload = _payload(args, schedule)
    payload["args"]["bridge_functional_weight"] = 0.0

    with pytest.raises(ValueError, match="bridge_functional_weight"):
        validate_resume_compatibility(payload, args, schedule, world_size=8)


def test_resume_rejects_changed_world_size() -> None:
    args = _args()
    schedule = _schedule(args)

    with pytest.raises(ValueError, match="world_size"):
        validate_resume_compatibility(_payload(args, schedule), args, schedule, world_size=2)
