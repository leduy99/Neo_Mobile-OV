from __future__ import annotations

from pathlib import Path


SCRIPT = Path("scripts/train_dreamlite_compact_v12_grounding_cascade_1node8gpu.sbatch")


def test_v12_launcher_uses_frozen_release_and_controlled_mixture() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "data/image_bridge_v1/releases/v12" in text
    assert "v12_data_preflight.json" in text
    assert "release_summary.json" in text
    assert "sha256(path) != expected_hash" in text
    assert 'PROMPT_WEIGHTS="0.7777777778,0.2222222222,0.0"' in text
    assert "--grounded-source-names grounded_cascade" in text
    assert "--grounded-batch-probability 0.10" in text
    assert "--semantic-prompt-probability 0" in text


def test_v12_launcher_preserves_v11_balanced_optimization_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-none}"' in text
    assert '--resume "${RESUME}"' in text
    assert "--representation-objective content_aware" in text
    assert "--functional-weight 5.0" in text
    assert "--functional-batch-size" in text
    assert "--student-state-probability 0.25" in text
    assert "--closed-loop-weight 0" in text
    assert 'SAVE_LATEST_EVERY:-5000' in text
    assert 'SAVE_ARCHIVE_EVERY:-20000' in text
