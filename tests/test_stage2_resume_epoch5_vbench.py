from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import test_stage2_resume_epoch4_vbench as helpers
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

runner, protocol, ROOT = helpers.runner, helpers.protocol, helpers.ROOT
SCRIPT = ROOT / "scripts/vbench_mobileov_stage2_resume_epoch5_1node8gpu.sbatch"


@pytest.fixture
def epoch5():
    with runner.evaluation_epoch(5):
        yield


def test_epoch_selection_restores_state_and_pins_distinct_archives():
    original = runner.EPOCH, runner.EPOCH_STEP, runner.SYSTEM, runner.DEFAULT_CHECKPOINT, runner.DEFAULT_OUTPUT
    with pytest.raises(RuntimeError, match="test cleanup"):
        with runner.evaluation_epoch(5):
            assert runner.EPOCH_STEP == 473295 and runner.SYSTEM == "stage2_resume_epoch5"
            assert runner.DEFAULT_CHECKPOINT.name == "stage2_dit_connector_step473295.pt"
            with runner.evaluation_epoch(4):
                assert runner.EPOCH_STEP == 378636
            assert runner.EPOCH_STEP == 473295
            raise RuntimeError("test cleanup")
    assert original == (runner.EPOCH, runner.EPOCH_STEP, runner.SYSTEM, runner.DEFAULT_CHECKPOINT, runner.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="pinned"):
        with runner.evaluation_epoch(3):
            pass


def test_cli_selects_epoch5_in_every_phase_without_mutating_default_epoch():
    for phase in ("run", "prepare", "generation-worker", "warm-scores", "score-worker", "summarize"):
        args = runner.parse_args(["--epoch", "5", "--phase", phase])
        assert args.epoch == 5 and args.workers == 8 and args.samples_per_prompt == 5
        if phase in ("run", "prepare"):
            assert args.checkpoint.name == "stage2_dit_connector_step473295.pt"
            assert args.output_dir.name == "vbench_mobileov_stage2_resume_epoch5_step473295_5x_dual_v2"
    assert runner.EPOCH == 4
    with pytest.raises(SystemExit):
        runner.parse_args(["--epoch", "6"])
    args = runner.parse_args(["--epoch", "5", "--max-prompts", "2", "--samples-per-prompt", "1"])
    assert args.output_dir.name.endswith("epoch5_step473295_1x_dual_v2_smoke2")


def test_epoch5_reader_requires_final_archive_not_epoch4_or_resume(tmp_path, epoch5):
    result = helpers.payload()
    path = tmp_path / "epoch5.pt"
    torch.save(result, path)
    with runner.epoch_policy():
        value, digest, target = runner.base.read_checkpoint(path)
    assert value["step"] == target == 473295 and digest == protocol.sha256(path)
    for wrong_step in (283977, 378636, 410000, 473296):
        wrong = dict(result, step=wrong_step)
        with pytest.raises(ValueError):
            runner.check_checkpoint(wrong, allow_incomplete=True)
    for forbidden in ("optimizer", "smolvlm2_lora"):
        with pytest.raises(ValueError):
            runner.check_checkpoint(dict(result, **{forbidden: {}}))


def test_epoch5_prepare_preserves_parent_and_marks_budget_complete(tmp_path, monkeypatch, epoch5):
    root, reference, _, _ = helpers.reference_fixture(tmp_path)
    before = protocol.sha256(root / "run_contract.json")
    contract = helpers.current_contract(reference)
    def prepare(args):
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(contract))))
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=True))
        return deepcopy(contract)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "epoch5", reference_run=root, max_prompts=0)
    actual = runner.prepare(args)
    assert runner.prepare(args) == actual
    assert actual["training_complete"] is True
    assert actual["epoch5_evaluation"]["expected_step"] == 473295
    assert actual["epoch5_evaluation"]["additional_updates"] == 189318
    assert "epoch4_evaluation" not in actual and "step473295" in actual["scope"]
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert protocol.sha256(root / "run_contract.json") == before
    assert runner.checked_contract(args.output_dir) == actual
    with runner.evaluation_epoch(4), pytest.raises(ValueError, match="epoch-4"):
        runner.checked_contract(args.output_dir)
    actual["training_complete"] = False
    protocol.atomic_json(args.output_dir / "run_contract.json", actual)
    with pytest.raises(ValueError, match="epoch-5"):
        runner.checked_contract(args.output_dir)


def test_epoch5_summary_compares_both_exports_and_all_dimensions(tmp_path, monkeypatch, epoch5):
    contract, parent, assets = helpers.prepared_fixture(tmp_path)
    output = tmp_path / "epoch5"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    def tabulate(folder):
        variant = "no_quicksr" if "no_quicksr" in folder.parts else "quicksr"
        score = deepcopy(parent["scores"][variant][runner.previous.PARENT_SYSTEM])
        score["total_score"] += .01 if variant == "no_quicksr" else .02
        return score
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = tabulate
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    for variant in runner.base.VARIANTS:
        for dimension in protocol.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                 f"{dimension}_eval_results.json", {dimension: [.5, []]})
    runner.summarize(SimpleNamespace(output_dir=output))
    report = protocol.read_json(output / "quicksr_ablation.json")
    assert report["status"] == "complete" and report["protocol"]["training_complete"] is True
    assert set(report["scores"]["quicksr"]) == {"stage2_resume_epoch5"}
    comparison = protocol.read_json(output / "epoch5_vs_epoch3.json")
    assert len(comparison["rows"]) == 19
    assert comparison["rows"]["total_score"]["no_quicksr"]["difference_pp"] == pytest.approx(1.)
    assert comparison["rows"]["total_score"]["quicksr"]["difference_pp"] == pytest.approx(2.)
    assert "Epoch 5 QuickSR" in (output / "epoch5_vs_epoch3.md").read_text()
    assert not (output / "epoch4_vs_epoch3.md").exists()


def test_all_subprocesses_keep_epoch5_and_eight_workers(tmp_path, monkeypatch):
    args = SimpleNamespace(epoch=5, phase="run", output_dir=tmp_path / "epoch5", workers=8,
                           max_prompts=0, vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare", lambda _: {})
    monkeypatch.setattr(runner, "verify_score_assets", lambda *a: None)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    calls = []
    def process(argv, **kwargs):
        assert option(argv, "--epoch") == "5"
        calls.append(option(argv, "--phase"))
    def workers(commands, devices, folder, **kwargs):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        for argv in commands:
            assert option(argv, "--epoch") == "5"
            assert option(argv, "--workers") == "8"
        calls.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert calls == ["warm-scores", "generation-worker", "score-worker", "summarize"]
    assert runner.EPOCH == 4


def test_epoch5_slurm_job_uses_final_weight_distinct_output_and_eight_gpus(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, env, run = launch
    shared = root / "scripts/vbench_mobileov_stage2_resume_epoch4_1node8gpu.sbatch"
    shared.write_text(helpers.SCRIPT.read_text())
    weight = root / "output/mobileov_stage2_resume5epochs16gpu/17892008/stage2_dit_connector_step473295.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"archived epoch5 weights")
    reference = root / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(reference / name, {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
for name in ("quicksr_ablation", "epoch5_vs_epoch3"):
    (output/(name+".json")).write_text("{}")
    (output/(name+".md")).write_text("mock epoch5 report")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--epoch") == "5"
    assert option(argv, "--checkpoint") == str(weight.relative_to(root))
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--output-dir").endswith("resume_epoch5_step473295_5x_dual_v2")
    assert "mock epoch5 report" in result.stdout
    assert "#SBATCH --gres=gpu:8" in SCRIPT.read_text() and "#SBATCH --nodes=1" in SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    Path(env["ARGV_LOG"]).unlink()
    weight.unlink()
    result, argv = run()
    assert result.returncode != 0 and not argv and "no latest/HF fallback" in result.stderr
