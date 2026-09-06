from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_dreamlite_openvid_step_following_full_1node8gpu.sbatch"


def test_full_launcher_uses_one_openvid_pass_and_all_call_kd() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "FULL_PASS_STEPS=$(( PROMPT_ROWS / GLOBAL_BATCH ))" in text
    assert "--functional-objective teacher_following" in text
    assert "--skip-representation-loss" in text
    assert "--functional-start-step 1" in text
    assert "--closed-loop-weight 0" in text
    assert "--grounded-batch-probability 0" in text
    assert "--init-bridge-checkpoint \"${V11_CKPT}\"" in text
    assert "dreamlite_compact_v11_v10_rates_from_scratch/17291461" in text


def test_full_launcher_has_resource_and_checkpoint_preflight() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --time=72:00:00" in text
    assert "Expected V11-balanced step 160000" in text
    assert "required = {\"caption_short\", \"caption_medium\", \"caption_long\"}" in text
    assert "run_contract.json" in text
