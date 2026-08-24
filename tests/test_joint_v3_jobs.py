from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT_SCRIPT = ROOT / "scripts" / "train_mobileov_monolithic_joint_v3_1node8gpu.sbatch"
DMD_SCRIPT = ROOT / "scripts" / "reproduce_neodragon_pyramidal_dmd_v3_1node8gpu.sbatch"


def test_joint_v3_is_a_controlled_step10k_continuation() -> None:
    script = JOINT_SCRIPT.read_text(encoding="utf-8")

    assert "neodragon_dit_bridge_step010000.pt" in script
    assert '--steps "${STEPS:-4000}"' in script
    assert '--lr "${DIT_LR:-1e-6}"' in script
    assert '--bridge-lr "${BRIDGE_LR:-1e-7}"' in script
    assert "--train-bridge" in script
    assert "--bridge-start-step 0" in script
    assert '--bridge-warmup-steps "${BRIDGE_WARMUP_STEPS:-500}"' in script
    assert "--flow-contract pyramid" in script
    assert "--stage-sampling-ratio 1,1,1" in script
    assert "--train-last-n-blocks 0" in script
    assert "--parallel fsdp" in script


def test_joint_v3_variants_differ_only_through_objective_args() -> None:
    script = JOINT_SCRIPT.read_text(encoding="utf-8")

    assert 'case "${VARIANT}" in' in script
    assert "distill|flow_only" in script
    assert "--objective-mode joint-distill" in script
    assert "--objective-mode flow-only" in script
    assert '--distill-weight "${DISTILL_WEIGHT:-1.0}"' in script
    assert '--cfg-distill-weight "${CFG_DISTILL_WEIGHT:-0.5}"' in script
    assert '--bridge-repr-weight "${BRIDGE_REPR_WEIGHT:-0.05}"' in script
    assert '--bridge-functional-weight "${BRIDGE_FUNCTIONAL_WEIGHT:-0.25}"' in script
    assert "OBJECTIVE_ARGS=(--objective-mode flow-only)" in script


def test_three_experiment_suite_has_bounded_checkpointing() -> None:
    joint = JOINT_SCRIPT.read_text(encoding="utf-8")
    dmd = DMD_SCRIPT.read_text(encoding="utf-8")

    assert '--save-every "${SAVE_EVERY:-500}"' in joint
    assert '--archive-every "${ARCHIVE_EVERY:-2000}"' in joint
    assert '--steps "${STEPS:-10000}"' in dmd
    assert "--rollout-aware-v3" in dmd
    assert "--no-include-first-unit" in dmd
