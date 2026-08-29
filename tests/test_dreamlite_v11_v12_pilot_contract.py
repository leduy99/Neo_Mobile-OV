from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_dreamlite_v11_plus_v12_grounded_pilot_1node8gpu.sbatch"


def test_pilot_changes_only_the_grounded_source_contract() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'TARGET_STEP="${TARGET_STEP:-20000}"' in script
    assert 'LR="${LR:-4e-6}"' in script
    assert 'SEMANTIC_PROBABILITY="${SEMANTIC_PROBABILITY:-0.12}"' in script
    assert 'GROUNDED_PROBABILITY="${GROUNDED_PROBABILITY:-0.10}"' in script
    assert 'PROMPT_WEIGHTS="0.7142857143,0.2857142857,0.0"' in script
    assert 'PROMPT_NAMES="journeydb,shortcaption,grounded_cascade"' in script
    assert "d2_grounded_high_precision_50k.csv" in script
    assert "--init-bridge-checkpoint" in script
    assert "--representation-objective content_aware" in script
    assert "--functional-start-step 1" in script
    assert "--student-state-start-step 1" in script
    assert "--training-version v11_plus_v12_grounded_pilot" in script
    assert 'SAVE_LATEST_EVERY:-1000' in script
    assert 'SAVE_ARCHIVE_EVERY:-5000' in script

    # The failed V12 broad/compositional replacement must not leak into this pilot.
    assert "d1_broad_train.csv" not in script
    assert "d2_compositional_train.csv" not in script
