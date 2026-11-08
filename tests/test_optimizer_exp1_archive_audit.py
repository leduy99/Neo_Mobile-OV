import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import audit_mobileov_optimizer_exp1 as audit


def write_training_run(root, world_size=16):
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(scope=audit.POLICY, steps=303977, world_size=world_size,
                    accumulation=4 if world_size == 16 else 8,
                    optimizer_exp1=dict(policy=audit.POLICY, parent_step=283977,
                                        global_batch=64, total_updates=20000))
    complete = dict(format=audit.POLICY, exp1_update=20000, step=303977, ema_updates=20000)
    manifest = dict(format=audit.POLICY, checkpoints={})
    for update in (10000, 20000):
        for variant in ("raw", "ema"):
            name = f"stage2_dit_connector_{variant}_update{update:06d}.pt"
            (root / name).write_bytes(b"fixture weights")
            manifest["checkpoints"][f"{variant}_{update}"] = dict(
                path=name, sha256="a" * 64, step=283977 + update,
                exp1_update=update, weight_variant=variant, ema_updates=update)
    for name, value in (("run_contract.json", contract), ("training_complete.json", complete),
                        ("evaluation_checkpoints.json", manifest)):
        (root / name).write_text(json.dumps(value))
    return root


@pytest.mark.parametrize("world_size", [8, 16])
def test_completed_run_admits_exactly_four_archives_without_loading_weights(tmp_path, world_size):
    root = write_training_run(tmp_path / "training", world_size)
    report = audit.audit_run(root, require_complete=True, expected_world_size=world_size)
    assert len(report["archives"]) == 4
    assert report["training_world_size"] * report["accumulation"] == 64
    assert report["completion"]["exp1_update"] == 20000
    assert "full_weight_SHA256_and_tensors_checked_by_evaluator" in report["verification"]


@pytest.mark.parametrize("case", ["no_complete", "wrong_final", "wrong_world", "no_manifest", "no_weight", "empty_weight", "bad_entry"])
def test_failed_or_wrong_run_is_rejected_before_gpu_generation(tmp_path, case):
    root = write_training_run(tmp_path / "training")
    if case == "no_complete":
        (root / "training_complete.json").unlink()
    elif case == "wrong_final":
        (root / "training_complete.json").write_text(json.dumps(dict(format=audit.POLICY, exp1_update=10000)))
    elif case == "wrong_world":
        value = json.loads((root / "run_contract.json").read_text())
        value.update(world_size=8, accumulation=8)
        (root / "run_contract.json").write_text(json.dumps(value))
    elif case == "no_manifest":
        (root / "evaluation_checkpoints.json").unlink()
    elif case in ("no_weight", "empty_weight"):
        path = root / "stage2_dit_connector_ema_update020000.pt"
        path.unlink() if case == "no_weight" else path.write_bytes(b"")
    else:
        value = json.loads((root / "evaluation_checkpoints.json").read_text())
        value["checkpoints"]["raw_10000"]["path"] = "../other.pt"
        (root / "evaluation_checkpoints.json").write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit_run(root, require_complete=True, expected_world_size=16)


def test_intermediate_archive_can_be_audited_without_claiming_completion(tmp_path):
    root = write_training_run(tmp_path / "training")
    (root / "training_complete.json").unlink()
    (root / "stage2_dit_connector_ema_update020000.pt").unlink()
    assert len(audit.audit_run(root, [10000])["archives"]) == 2


def test_cli_requires_no_torch_av_or_transformers_and_preserves_training_files(tmp_path):
    root = write_training_run(tmp_path / "training")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    output = tmp_path / "eval/training_audit.json"
    source = f"""import runpy, sys
sys.modules.update(torch=None, av=None, transformers=None)
sys.argv = ['audit', '--run', sys.argv[1], '--require-complete', '--output', sys.argv[2]]
runpy.run_path({str(Path(audit.__file__).resolve())!r}, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", source, str(root), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["completion"]["step"] == 303977
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    failed = subprocess.run([sys.executable, str(Path(audit.__file__)), "--run", str(root),
                             "--output", str(root / "overwrite.json")], capture_output=True, text=True)
    assert failed.returncode != 0 and not (root / "overwrite.json").exists()
