from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "scripts"
    / "vbench_dreamlite_v11_plus_v12_pilot20k_exp1_64k_1node1gpu.sbatch"
)
COMMON = ROOT / "scripts" / "vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch"


def test_pilot_vbench_matches_the_v11_v12_one_video_protocol() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"' in script
    assert "dreamlite_v11_plus_v12_grounded_pilot/17421424" in script
    assert 'IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-20000}"' in script
    assert "v11_plus_v12_grounded_pilot" in script
    assert 'IMAGE_EXPECTED_SOURCE_STEP="${IMAGE_EXPECTED_SOURCE_STEP:-160000}"' in script
    assert "v11_balanced" in script
    assert 'IMAGE_EXPECTED_BRIDGE_TENSORS="${IMAGE_EXPECTED_BRIDGE_TENSORS:-56}"' in script
    assert "neo_exp1_bridge_functional/17108893" in script
    assert 'WITH_QUICKSR="${WITH_QUICKSR:-0}"' in script
    assert "unset RECAPTION_FILE" in script
    assert "source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch" in script


def test_common_vbench_supports_strict_optional_image_provenance() -> None:
    script = COMMON.read_text(encoding="utf-8")

    for field in (
        "IMAGE_EXPECTED_TRAINING_VERSION",
        "IMAGE_EXPECTED_SOURCE_STEP",
        "IMAGE_EXPECTED_SOURCE_TRAINING_VERSION",
        "IMAGE_EXPECTED_BRIDGE_TENSORS",
    ):
        assert field in script
    assert "Image checkpoint provenance mismatch" in script
