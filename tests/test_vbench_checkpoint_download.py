from pathlib import Path


def test_vbench_download_recovers_from_corrupt_shared_checkpoint() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch"
    ).read_text(encoding="utf-8")

    assert "flock -x 9" in script
    assert 'if ! actual_step=$("${PYTHON_BIN}"' in script
    assert 'echo "Replacing unreadable checkpoint ${target}"' in script
    assert "force_download=True" in script
    assert 'f".tmp.{os.getpid()}"' in script
