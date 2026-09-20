import copy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from new_mobile_ov.training import stage1_alignment as data
from new_mobile_ov.training.stage2_alignment import FORMAT
from new_mobile_ov.training import stage2_alignment as stage2
from tools.train_mobileov_stage1 import generator_for, parse_args, validate_resume


def fixture_payload():
    contract = dict(format=FORMAT, steps=100000, lr=2e-5, accumulation=2, world_size=8,
                    seed=20260911, data_summary_sha256="data", warmup=50, vae_window_size=8)
    saved = dict(format=FORMAT, contract=contract, step=100000, connector_spec={"dim": 8},
                 connector={"weight": torch.ones(1)}, dit={"weight": torch.ones(1)},
                 optimizer={"state": {0: {"step": torch.tensor(100000.)}}, "param_groups": [{"params": [0]}]})
    return saved, dict(contract, steps=150000)


def test_extension_requires_opt_in_and_only_changes_target():
    saved, target = fixture_payload()
    with pytest.raises(ValueError, match="Resume contract mismatch.*steps"):
        validate_resume(saved, target, saved["connector_spec"])
    before = copy.deepcopy(saved)
    assert validate_resume(saved, target, saved["connector_spec"], extend_steps=True, expected_step=100000) == 100000
    assert saved["contract"] == before["contract"]
    assert torch.equal(saved["optimizer"]["state"][0]["step"], before["optimizer"]["state"][0]["step"])
    saved["contract"]["steps"] = 150000
    saved["step"] = 120000
    assert validate_resume(saved, target, saved["connector_spec"], expected_step=120000) == 120000


@pytest.mark.parametrize("key,value", [("lr", 1e-5), ("world_size", 2), ("accumulation", 4),
                                      ("seed", 1), ("data_summary_sha256", "other"),
                                      ("warmup", 500), ("vae_window_size", 16), ("steps", 90000)])
def test_extension_rejects_other_recipe_changes(key, value):
    saved, target = fixture_payload()
    target[key] = value
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        validate_resume(saved, target, saved["connector_spec"], extend_steps=True)


@pytest.mark.parametrize("change,match", [("optimizer", "optimizer state"), ("empty_optimizer", "optimizer state"),
                                         ("format", "format mismatch"), ("connector_spec", "architecture mismatch"),
                                         ("dit", "trained weights"), ("step", "Expected resume step"),
                                         ("future_step", "invalid step")])
def test_extension_rejects_bad_resume(change, match):
    saved, target = fixture_payload()
    if change in ("optimizer", "dit"):
        saved.pop(change)
    elif change == "empty_optimizer":
        saved["optimizer"]["state"] = {}
    elif change == "connector_spec":
        saved[change] = {"dim": 99}
    elif change == "format":
        saved[change] = "stage1"
    else:
        saved["step"] = 99999 if change == "step" else 100001
    with pytest.raises(ValueError, match=match):
        validate_resume(saved, target, {"dim": 8}, extend_steps=True, expected_step=100000)


@pytest.mark.parametrize("extra", [["--extend-steps"], ["--expected-resume-step", "100000"],
                                  ["--resume", "r.pt", "--expected-resume-step", "-1"],
                                  ["--resume", "r.pt", "--expected-resume-step", "150000"]])
def test_invalid_extension_arguments(extra):
    with pytest.raises(SystemExit):
        parse_args(["--phase", "2", "--init-connector", "init.pt", "--output-dir", "out", "--steps", "150000", *extra])


def test_100k_extension_keeps_sample_and_noise_streams_on_all_eight_ranks(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "sources.json").write_text("{}")
    for task in ("t2i", "t2v"):
        (tmp_path / "train" / f"{task}.jsonl").write_text("".join(
            json.dumps(dict(task=task, sample_id=f"{task}:{i}")) + "\n" for i in range(101)))
    monkeypatch.setattr(data, "prepare_sample", lambda record, *args: record.copy())
    for rank in range(8):
        common = dict(steps=150000, accumulation=2, world_size=8, rank=rank, seed=20260911, tasks=("t2i", "t2v"))
        full = data.Stage1Dataset(tmp_path, **common)
        extended = data.Stage1Dataset(tmp_path, start_step=100000, **common)
        assert len(extended) == 100000
        for index in (0, 1, 2, 201, 99998, 99999):
            a, b = extended[index], full[200000 + index]
            assert a == b
            assert torch.equal(torch.randn(8, generator=generator_for("cpu", 20260911, a["micro"], rank)),
                               torch.randn(8, generator=generator_for("cpu", 20260911, b["micro"], rank)))


def launcher_env(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = dict(os.environ, PROJECT_ROOT=str(root), CONDA_ENV=str(tmp_path), SLURM_JOB_ID="999",
               OUT=str(tmp_path / "out"), DATA_ROOT=str(tmp_path / "data"), TMPDIR="/tmp",
               HF_HOME=str(tmp_path / "hf"), PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(tmp_path / "calls"))
    # Stale shell settings must not silently change the continuation recipe.
    env.update(STEPS="5000", LR=".1", ACCUMULATION="3", SAVE_EVERY="1", EXPECTED_RESUME_STEP="100000")
    for key, name in (("INIT_CONNECTOR", "init.pt"), ("RESUME", "source/stage2_resume.pt")):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("checkpoint")
        env[key] = str(path)
    Path(env["DATA_ROOT"]).mkdir()
    (Path(env["DATA_ROOT"]) / ".stage1_data_complete").write_text("ready")
    Path(env["OUT"]).mkdir()
    for name, body in {
        "module": "exit 0",
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s STEP=%s OUT=%s\\n" "$*" "$RESUME" "$EXPECTED_RESUME_STEP" "$OUT" >> "$CALLS"',
    }.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    return env, root / "scripts/train_mobileov_stage2_extend150k_1node8gpu.sbatch"


def test_extension_launcher_recipe_and_pause_resubmission(tmp_path):
    env, script = launcher_env(tmp_path)
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    calls = Path(env["CALLS"])
    text = calls.read_text()
    for expected in ("--phase 2", "--steps 150000", "--accumulation 2", "--lr 2e-5", "--warmup 50",
                     "--save-every 10000 --archive-every 10000", "--validate-every 1000", "--validation-samples 2",
                     "--extend-steps", "--expected-resume-step 100000", f"--resume {env['RESUME']}"):
        assert expected in text
    assert "CONTINUE" not in text
    out = Path(env["OUT"])
    (out / "training_paused.json").write_text('{"step": 123456}')
    (out / "stage2_resume.pt").write_text("checkpoint")
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert f"CONTINUE --dependency=afterok:999 --export=ALL" in calls.read_text()
    assert f"RESUME={out}/stage2_resume.pt STEP=123456 OUT={out}" in calls.read_text()
    (out / "training_complete.json").write_text('{"step": 150000}')
    calls.unlink()
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert "CONTINUE" not in calls.read_text()


def test_extension_launcher_missing_resume_and_source_overwrite_fail_before_torchrun(tmp_path):
    env, script = launcher_env(tmp_path)
    result = subprocess.run(["bash", str(script)], env=dict(env, OUT=str(Path(env["RESUME"]).parent)), capture_output=True, text=True)
    assert result.returncode != 0 and "new OUT directory" in result.stderr
    Path(env["RESUME"]).unlink()
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Missing optimizer checkpoint" in result.stderr
    assert not Path(env["CALLS"]).exists()


def test_extension_launcher_rejects_stale_resume_in_an_existing_output(tmp_path):
    env, script = launcher_env(tmp_path)
    out = Path(env["OUT"])
    (out / "run_contract.json").write_text('{"steps": 150000}')
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "OUT already contains a run" in result.stderr
    assert not Path(env["CALLS"]).exists()
    own_resume = out / "stage2_resume.pt"
    own_resume.write_text("checkpoint")
    result = subprocess.run(["bash", str(script)], env=dict(env, RESUME=str(own_resume), EXPECTED_RESUME_STEP="120000"),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--expected-resume-step 120000" in Path(env["CALLS"]).read_text()


def test_local_extension_smoke_is_slurm_only_and_syntactically_valid():
    script = Path(__file__).resolve().parents[1] / "scripts/smoke_mobileov_stage2_extension_local.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "#SBATCH --gres=gpu:2" in text
    assert "--expected-resume-step 12 --extend-steps --stop-after 14" in text
    assert "--expected-resume-step 14" in text


@pytest.mark.parametrize("extra", [["--reconstruction-weight", "-1"], ["--reconstruction-weight", "nan"],
                                  ["--reconstruction-weight", "inf"], ["--enable-reconstruction-on-resume"],
                                  ["--reconstruction-weight", ".1", "--enable-reconstruction-on-resume"]])
def test_invalid_reconstruction_arguments(extra):
    with pytest.raises(SystemExit):
        parse_args(["--phase", "2", "--init-connector", "init.pt", "--output-dir", "out", *extra])


def test_reconstruction_is_opt_in_and_does_not_change_sampling_defaults():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "out", "--reconstruction-weight", ".1"])
    args = parse_args(["--phase", "2", "--init-connector", "init.pt", "--output-dir", "out",
                       "--reconstruction-weight", ".1"])
    assert args.tasks == ("t2i", "t2v") and args.accumulation == 2 and args.lr == 2e-5
    assert not args.enable_reconstruction_on_resume
    plain = parse_args(["--phase", "2", "--init-connector", "init.pt", "--output-dir", "out"])
    assert plain.reconstruction_weight == 0


def test_reconstruction_migration_preserves_every_other_contract_field():
    saved, target = fixture_payload()
    target["auxiliary_reconstruction"] = stage2.reconstruction_contract(.1)
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        validate_resume(saved, target, saved["connector_spec"], extend_steps=True)
    kwargs = dict(extend_steps=True, enable_reconstruction=True, expected_step=100000)
    assert validate_resume(saved, target, saved["connector_spec"], **kwargs) == 100000
    for change in (dict(lr=1e-5), dict(accumulation=3), dict(world_size=2), dict(data_summary_sha256="changed")):
        with pytest.raises(ValueError, match="Resume contract mismatch"):
            validate_resume(saved, dict(target, **change), saved["connector_spec"], **kwargs)
    saved["contract"] = copy.deepcopy(target)
    saved["step"] = 110000
    assert validate_resume(saved, target, saved["connector_spec"], expected_step=110000) == 110000
    changed_weight = dict(target, auxiliary_reconstruction=stage2.reconstruction_contract(.2))
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        validate_resume(saved, changed_weight, saved["connector_spec"], enable_reconstruction=True)
    target.pop("auxiliary_reconstruction")
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        validate_resume(saved, target, saved["connector_spec"], enable_reconstruction=True)


@pytest.mark.parametrize("accumulation", [2, 4, 6])
def test_reconstruction_weight_is_not_divided_by_the_video_batches(accumulation):
    coefficient = stage2.reconstruction_coefficient(.1, accumulation)
    assert coefficient * (accumulation // 2) == pytest.approx(.1)
    assert 1 / accumulation * (accumulation // 2) == .5


def test_reconstruction_launcher_passes_weight_and_resubmits_its_own_recipe(tmp_path):
    env, _ = launcher_env(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/train_mobileov_stage2_reconstruction150k_1node8gpu.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    calls = Path(env["CALLS"])
    text = calls.read_text()
    for arg in ("--phase 2", "--steps 150000", "--accumulation 2", "--lr 2e-5", "--reconstruction-weight 0.1",
                "--enable-reconstruction-on-resume", "--expected-resume-step 100000", "--extend-steps"):
        assert arg in text
    out = Path(env["OUT"])
    (out / "training_paused.json").write_text('{"step": 123456}')
    (out / "stage2_resume.pt").write_text("checkpoint")
    calls.unlink()
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert "CONTINUE --dependency=afterok:999 --export=ALL scripts/train_mobileov_stage2_reconstruction150k_1node8gpu.sbatch" in calls.read_text()
    assert "STEP=123456" in calls.read_text()
