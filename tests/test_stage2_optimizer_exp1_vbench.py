from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import benchmark_mobileov_stage2_optimizer_exp1_dual as runner
from tools import vbench_joint_native_protocol as protocol
import test_stage2_media3epochs_dual_vbench as helpers
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

ROOT = Path(__file__).resolve().parents[1]


def payload(update=10000, variant="ema"):
    value = helpers.payload()
    value.update(step=283977 + update, exp1_update=update, weight_variant=variant, ema_updates=update)
    value["contract"].update(scope=runner.POLICY, world_size=16, accumulation=4, warmup=0, lr=2e-5,
        optimizer_exp1=dict(policy=runner.POLICY, parent_step=283977, global_batch=64, total_updates=20000),
        ema=dict(decay=.9995), lr_schedule=dict(decay_steps=20000), parent_contract=dict(steps=283977),
        auxiliary_reconstruction=dict(weight=.1))
    return value


@pytest.mark.parametrize("update,variant", [(10000, "raw"), (10000, "ema"), (20000, "raw"), (20000, "ema")])
@pytest.mark.parametrize("world_size,accumulation", [(16, 4), (8, 8)])
def test_snapshot_reader_requires_exact_raw_or_ema_and_never_uses_latest(tmp_path, update, variant,
                                                                      world_size, accumulation):
    path = tmp_path / "snapshot.pt"
    candidate = payload(update, variant)
    candidate["contract"].update(world_size=world_size, accumulation=accumulation)
    torch.save(candidate, path)
    before = runner.base.SYSTEM, runner.base.check_media_checkpoint
    with runner.case_policy(update, variant):
        value, digest, target = runner.base.read_checkpoint(path)
        assert target == value["step"] == 283977 + update and digest == protocol.sha256(path)
        assert runner.base.SYSTEM == f"stage2_exp1_{variant}_{update}"
        with pytest.raises(ValueError, match="SHA256"):
            runner.base.read_checkpoint(path, expected_sha="0" * 64)
    assert before == (runner.base.SYSTEM, runner.base.check_media_checkpoint)


@pytest.mark.parametrize("world_size,accumulation", [(8, 4), (16, 8), (4, 16)])
def test_snapshot_reader_rejects_wrong_training_layout(world_size, accumulation):
    candidate = payload()
    candidate["contract"].update(world_size=world_size, accumulation=accumulation)
    with pytest.raises(ValueError):
        runner.check_checkpoint(candidate, 10000, "ema")


@pytest.mark.parametrize("change", ["variant", "update", "ema", "batch", "history", "optimizer", "nan", "loss"])
def test_wrong_snapshot_is_rejected(change):
    value = payload()
    if change == "variant": value["weight_variant"] = "raw"
    elif change == "update": value["exp1_update"] = 20000
    elif change == "ema": value["ema_updates"] = 0
    elif change == "batch": value["contract"]["optimizer_exp1"]["global_batch"] = 16
    elif change == "history": value["contract"]["history_corruption"] = {}
    elif change == "optimizer": value["optimizer"] = {}
    elif change == "loss": value["contract"]["auxiliary_reconstruction"]["weight"] = .2
    else: value["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.check_checkpoint(value, 10000, "ema")


def test_prepare_pins_manifest_hash_and_updates_sample_contract(tmp_path, monkeypatch):
    train = tmp_path / "train"
    train.mkdir()
    path = train / "ema_10k.pt"
    torch.save(payload(), path)
    protocol.atomic_json(train / "evaluation_checkpoints.json", dict(format=runner.POLICY, checkpoints={
        "ema_10000": dict(path=path.name, sha256=protocol.sha256(path), exp1_update=10000,
                          step=293977, weight_variant="ema")}))
    contract = helpers.contract()
    def prepare(args):
        assert args.checkpoint == path and args.expected_sha256 == protocol.sha256(path)
        runner.base.read_checkpoint(path)
        result = dict(contract, system=runner.base.SYSTEM, training_contract=payload()["contract"])
        for name, value in (("vbench_info.json", []), ("checkpoint_audit.json", {}), ("sample_plan.json", dict(samples=[]))):
            protocol.atomic_json(args.output_dir / name, value)
        return result
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(exp1_run=train, output_dir=tmp_path / "eval", reference_run=tmp_path / "parent")
    child, result = runner.prepare_case(args, 10000, "ema", None)
    assert result["exp1_evaluation"]["additional_updates"] == 10000
    assert not result["exp1_evaluation"]["experiment_complete"]
    assert runner.worker_policy(child.output_dir) == (10000, "ema")
    assert protocol.read_json(child.output_dir / "sample_plan.json")["contract_sha256"] == protocol.fingerprint(result)
    with pytest.raises(ValueError, match="Missing pinned"):
        runner.prepare_case(args, 20000, "ema", None)


def test_sweep_summary_has_both_exports_and_three_scores(tmp_path):
    system = "stage2_exp1_ema_10000"
    contract = dict(system=system)
    values = dict(quality_score=.8, semantic_score=.6, total_score=.76)
    scores = {variant: {system: values} for variant in runner.base.VARIANTS}
    child = tmp_path / "ema_10000"
    protocol.atomic_json(child / "quicksr_ablation.json", dict(status="complete", protocol=contract, scores=scores))
    reference = ({}, dict(scores={v: {runner.previous.PARENT_SYSTEM: dict(values, total_score=.75)}
                                 for v in runner.base.VARIANTS}), [])
    runner.summarize_sweep(SimpleNamespace(output_dir=tmp_path), [(child, contract)], reference)
    rows = protocol.read_json(tmp_path / "comparison.json")["rows"]
    assert {row["output"] for row in rows} == {"quicksr", "no_quicksr"}
    assert rows[0]["quality_score"] == 80 and rows[0]["semantic_score"] == 60
    assert rows[0]["total_vs_parent_pp"] == pytest.approx(1.)


def test_full_sweep_runs_four_cases_with_eight_workers_each(tmp_path, monkeypatch):
    args = runner.parse_args(["--exp1-run", str(tmp_path / "train"), "--output-dir", str(tmp_path / "eval"),
                             "--vbench-repo", str(tmp_path), "--vbench-python", sys.executable])
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner.previous, "read_reference", lambda _: ({}, {}, []))
    monkeypatch.setattr(runner, "verify_score_assets", lambda *a: None)
    def prepare(args, update, variant, reference):
        child = deepcopy(args)
        child.output_dir = args.output_dir / f"{variant}_{update}"
        contract = dict(system=f"stage2_exp1_{variant}_{update}",
                        exp1_evaluation=dict(additional_updates=update, weights=variant))
        protocol.atomic_json(child.output_dir / "run_contract.json", contract)
        return child, contract
    monkeypatch.setattr(runner, "prepare_case", prepare)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    calls = []
    def process(argv, **kw):
        assert argv[1].endswith("benchmark_mobileov_stage2_optimizer_exp1_dual.py")
        calls.append(option(argv, "--phase"))
    def workers(commands, devices, folder, **kw):
        assert len(commands) == len(devices) == 8
        assert all(argv[1].endswith("benchmark_mobileov_stage2_optimizer_exp1_dual.py") for argv in commands)
        calls.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    monkeypatch.setattr(runner, "summarize_sweep", lambda args, cases, reference: calls.append(f"summary{len(cases)}"))
    runner.main()
    assert calls == ["warm-scores", "generation-worker", "score-worker", "summarize"] * 4 + ["summary4"]


def test_scoring_environment_does_not_import_av():
    code = '''import sys
sys.modules['av'] = None
from tools import benchmark_mobileov_stage2_optimizer_exp1_dual as runner
runner.base.core.warm_scores = lambda args: print('isolated-scores-ok')
sys.argv = ['eval', '--phase', 'warm-scores', '--output-dir', '/tmp/exp1-score-import']
runner.main()
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "isolated-scores-ok" in result.stdout


def test_eval_slurm_job_passes_explicit_run_and_both_milestones(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", ROOT / "scripts/vbench_mobileov_stage2_optimizer_exp1_1node8gpu.sbatch")
    root, env, run = launch
    train = root / "train output"
    protocol.atomic_json(train / "evaluation_checkpoints.json", {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ['ARGV_LOG']).write_text(json.dumps(argv))
out=pathlib.Path(argv[argv.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
(out/'comparison.md').write_text('mock Exp1 comparison')
''')
    result, argv = run(EXP1_RUN=str(train))
    assert result.returncode == 0, result.stderr
    assert option(argv, "--exp1-run") == str(train)
    assert argv[argv.index("--updates") + 1:argv.index("--reference-run")] == ["10000", "20000"]
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
