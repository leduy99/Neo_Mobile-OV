from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import benchmark_mobileov_stage2_history_fresh_epochs_dual as runner
from new_mobile_ov.training.stage2_history import MAX_HISTORY_SIGMA, history_contract
import test_stage2_resume_epoch4_vbench as clean
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

protocol, ROOT = clean.protocol, clean.ROOT


def payload(epoch):
    value = clean.payload()
    value["step"] = runner.EPOCH_STEPS[epoch]
    value["contract"].update(
        scope="fresh_multimodal_stage2_with_history_corruption; public_data; not_exact_UniVideo_recipe",
        history_corruption=history_contract(MAX_HISTORY_SIGMA),
        video_units="T2V/reference:uniform0_to6; anchor:uniform1_to6; teacher_forced_corrupted_history")
    return value


def current_contract(reference, epoch):
    result = deepcopy(reference)
    result.update(system=runner.system(epoch), training_complete=False, training_contract=payload(epoch)["contract"],
        checkpoint=dict(path=str(runner.epoch_paths(epoch)[0]), sha256="a" * 64,
                        step=runner.EPOCH_STEPS[epoch], final_step=runner.TARGET_STEP))
    return result


def prepared_fixture(tmp_path, epoch):
    root, reference, report, assets = clean.reference_fixture(tmp_path)
    result = current_contract(reference, epoch)
    result["history_fresh_evaluation"] = dict(
        training_run="17892009", epoch=epoch, expected_step=runner.EPOCH_STEPS[epoch],
        planned_final_step=runner.TARGET_STEP, history_corruption_at_inference=False,
        reference_run=str(root), reference_verified=True,
        reference_contract_sha256=protocol.fingerprint(reference),
        reference_report_sha256=protocol.fingerprint(report))
    return result, report, assets


@pytest.mark.parametrize("epoch", [3, 4])
def test_cli_pins_fresh_history_archives_and_distinct_outputs(epoch):
    args = runner.parse_args(["--epoch", str(epoch)])
    assert args.checkpoint == runner.epoch_paths(epoch)[0]
    assert "17892009" in str(args.checkpoint) and "17892008" not in str(args.checkpoint)
    assert args.checkpoint.name == f"stage2_dit_connector_step{runner.EPOCH_STEPS[epoch]}.pt"
    assert args.output_dir == runner.epoch_paths(epoch)[1]
    assert f"history_fresh_epoch{epoch}" in args.output_dir.name
    assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
    smoke = runner.parse_args(["--epoch", str(epoch), "--max-prompts", "2", "--samples-per-prompt", "1"])
    assert smoke.output_dir.name.endswith("_1x_dual_smoke2")
    for flags in ([], ["--epoch", "5"], ["--epoch", str(epoch), "--samples-per-prompt", "1"],
                  ["--epoch", str(epoch), "--allow-incomplete-checkpoint"]):
        with pytest.raises(SystemExit):
            runner.parse_args(flags)


@pytest.mark.parametrize("epoch", [3, 4])
def test_reader_accepts_exact_epoch_only_and_restores_legacy_policy(tmp_path, epoch):
    path = tmp_path / "archive.pt"
    torch.save(payload(epoch), path)
    before = runner.base.check_media_checkpoint, runner.base.SYSTEM
    with pytest.raises(ValueError, match="three-media-epoch"):
        runner.base.read_checkpoint(path)
    with runner.history_policy(epoch):
        value, digest, target = runner.base.read_checkpoint(path)
        assert value["step"] == runner.EPOCH_STEPS[epoch] and target == 473295
        assert digest == protocol.sha256(path) and runner.base.SYSTEM == runner.system(epoch)
        with pytest.raises(ValueError, match="SHA256 changed"):
            runner.base.read_checkpoint(path, expected_sha="0" * 64)
        with pytest.raises(ValueError):
            runner.check_checkpoint(value, 7 - epoch, allow_incomplete=True)
    assert before == (runner.base.check_media_checkpoint, runner.base.SYSTEM)


@pytest.mark.parametrize("change", ["step", "clean_resume", "history", "history_sigma", "fork", "lora", "optimizer",
                                    "lr", "reconstruction", "batch", "rng", "data", "tasks", "nan", "init_step"])
def test_wrong_run_recipe_or_weights_rejected_even_for_smoke(change):
    value = payload(3)
    training = value["contract"]
    if change == "step": value["step"] = 290000
    elif change == "clean_resume": value = clean.payload()
    elif change == "history": training.pop("history_corruption")
    elif change == "history_sigma": training["history_corruption"]["max_sigma"] = .2
    elif change == "fork": training["initialization"] = "stage2_weights_only_fork_fresh_optimizer"
    elif change == "lora": value["smolvlm2_lora"] = {}
    elif change == "optimizer": value["optimizer"] = {}
    elif change == "lr": training["lr"] = 1e-4
    elif change == "reconstruction": training["auxiliary_reconstruction"]["weight"] = .2
    elif change == "batch": training["accumulation"] = 2
    elif change == "rng": training["media_epochs"]["rng_layout"]["world_size"] = 16
    elif change == "data": training["media_epochs"]["videos"] = 83000
    elif change == "tasks": training["tasks"] = ["t2v"]
    elif change == "init_step": training["init_step"] = 283977
    else: value["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.check_checkpoint(value, 3, allow_incomplete=True)


@pytest.mark.parametrize("epoch", [3, 4])
def test_protocol_preserves_initialization_data_and_numerical_evaluation(tmp_path, epoch):
    _, reference, _, _ = clean.reference_fixture(tmp_path)
    current = current_contract(reference, epoch)
    assert runner.compare_protocol(reference, current)["policy"] == "identical_trainer_source"
    assert "parent_contract" not in current["training_contract"]
    for key in ("data_summary_sha256", "initial_weights_sha256", "init_step", "lr", "frozen_weights_sha256"):
        wrong = deepcopy(current)
        wrong["training_contract"][key] = "changed"
        with pytest.raises(ValueError):
            runner.compare_protocol(reference, wrong)
    for key in ("seed", "exports", "guidance", "library_versions", "implementation_sha256"):
        wrong = deepcopy(current)
        wrong[key] = "changed"
        with pytest.raises((ValueError, TypeError, AttributeError)):
            runner.compare_protocol(reference, wrong)


@pytest.mark.parametrize("epoch", [3, 4])
def test_prepare_is_separate_idempotent_and_not_a_parent_fork(tmp_path, monkeypatch, epoch):
    root, reference, _, _ = clean.reference_fixture(tmp_path)
    before = protocol.sha256(root / "run_contract.json")
    current = current_contract(reference, epoch)
    def prepare(args):
        assert runner.base.SYSTEM == runner.system(epoch)
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=False))
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(current))))
        return deepcopy(current)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(epoch=epoch, output_dir=tmp_path / f"history_ep{epoch}", reference_run=root, max_prompts=0)
    contract = runner.prepare(args)
    assert runner.prepare(args) == contract
    assert contract["history_fresh_evaluation"]["epoch"] == epoch
    assert contract["history_fresh_evaluation"]["training_run"] == "17892009"
    assert contract["history_fresh_evaluation"]["history_corruption_at_inference"] is False
    assert contract["training_complete"] is False
    assert "NOT the trained epoch-3 model" in contract["scope"]
    assert "parent_contract" not in contract["training_contract"]
    assert "epoch4_evaluation" not in contract and "continuation_evaluation" not in contract
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert protocol.sha256(root / "run_contract.json") == before
    assert runner.checked_contract(args.output_dir, epoch) == contract
    with pytest.raises(ValueError, match="pinned fresh-history"):
        runner.checked_contract(args.output_dir, 7 - epoch)
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(SimpleNamespace(**dict(vars(args), output_dir=root)))


@pytest.mark.parametrize("epoch", [3, 4])
def test_generation_reuses_numerical_engine_with_the_pinned_history_weights(tmp_path, monkeypatch, epoch):
    contract, _, _ = prepared_fixture(tmp_path, epoch)
    protocol.atomic_json(tmp_path / "run_contract.json", contract)
    path = tmp_path / "archive.pt"
    torch.save(payload(epoch), path)
    calls = []
    def worker(args, *, device=None):
        value, _, target = runner.base.read_checkpoint(path)
        calls.append((value["step"], target, runner.base.SYSTEM))
    monkeypatch.setattr(runner.base, "generation_worker", worker)
    runner.generation_worker(SimpleNamespace(epoch=epoch, output_dir=tmp_path))
    assert calls == [(runner.EPOCH_STEPS[epoch], 473295, runner.system(epoch))]


@pytest.mark.parametrize("epoch", [3, 4])
def test_summary_compares_all_dimensions_raw_and_quicksr_to_clean_epoch3(tmp_path, monkeypatch, epoch):
    contract, reference, assets = prepared_fixture(tmp_path, epoch)
    output = tmp_path / f"history_ep{epoch}"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    module = ModuleType("evaluate_vbench_resume")
    def tabulate(folder):
        variant = "no_quicksr" if "no_quicksr" in folder.parts else "quicksr"
        score = deepcopy(reference["scores"][variant][runner.previous.PARENT_SYSTEM])
        score["total_score"] += .01 if variant == "no_quicksr" else .02
        return score
    module.tabulate = tabulate
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    for variant in runner.base.VARIANTS:
        for dimension in protocol.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                 f"{dimension}_eval_results.json", {dimension: [.5, []]})
    runner.summarize(SimpleNamespace(epoch=epoch, output_dir=output))
    report = protocol.read_json(output / "quicksr_ablation.json")
    assert report["status"] == "complete" and report["protocol"]["training_complete"] is False
    assert set(report["scores"]["quicksr"]) == {runner.system(epoch)}
    comparison = protocol.read_json(output / "history_fresh_vs_clean_epoch3.json")
    assert len(comparison["rows"]) == 19
    assert comparison["rows"]["total_score"]["no_quicksr"]["difference_pp"] == pytest.approx(1.)
    assert comparison["rows"]["total_score"]["quicksr"]["difference_pp"] == pytest.approx(2.)
    assert f"History Ep{epoch} QuickSR" in (output / "history_fresh_vs_clean_epoch3.md").read_text()
    protocol.atomic_json(output / "score_assets.json", [dict(assets[0], sha256="changed")])
    with pytest.raises(ValueError, match="evaluation weights"):
        runner.verify_score_assets(output, contract)


@pytest.mark.parametrize("epoch", [3, 4])
def test_orchestration_propagates_epoch_to_all_eight_generation_and_scoring_workers(tmp_path, monkeypatch, epoch):
    args = SimpleNamespace(epoch=epoch, phase="run", output_dir=tmp_path / f"history_ep{epoch}", workers=8,
                           max_prompts=0, vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare", lambda _: {})
    monkeypatch.setattr(runner, "verify_score_assets", lambda *a: None)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    calls = []
    def process(argv, **kwargs):
        assert option(argv, "--epoch") == str(epoch)
        calls.append(option(argv, "--phase"))
    def workers(commands, devices, folder, **kwargs):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        for argv in commands:
            assert option(argv, "--epoch") == str(epoch) and option(argv, "--workers") == "8"
            assert "history_fresh_epochs_dual.py" in argv[1]
        calls.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert calls == ["warm-scores", "generation-worker", "score-worker", "summarize"]


@pytest.mark.parametrize("epoch", [3, 4])
def test_berzelius_jobs_have_distinct_short_names_logs_and_outputs(launch, monkeypatch, epoch):
    script = ROOT / f"scripts/vbench_mobileov_stage2_history_fresh_epoch{epoch}_1node8gpu.sbatch"
    monkeypatch.setattr(launch_helpers, "SCRIPT", script)
    root, env, run = launch
    (root / "scripts/run_vbench_mobileov_history_fresh_epoch.sh").write_text(
        (ROOT / "scripts/run_vbench_mobileov_history_fresh_epoch.sh").read_text())
    weight = root / "output/mobileov_stage2_history_fresh5epochs16gpu/17892009" / f"stage2_dit_connector_step{runner.EPOCH_STEPS[epoch]}.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fresh history archive")
    reference = root / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(reference / name, {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
for name in ("quicksr_ablation", "history_fresh_vs_clean_epoch3"):
    (output/(name+".json")).write_text("{}")
    (output/(name+".md")).write_text("mock fresh-history report")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--epoch") == str(epoch)
    assert option(argv, "--checkpoint") == str(weight.relative_to(root))
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--seed") == "2026" and "--allow-incomplete-checkpoint" not in argv
    assert option(argv, "--output-dir").endswith(f"history_fresh_epoch{epoch}_step{runner.EPOCH_STEPS[epoch]}_5x_dual")
    assert "mock fresh-history report" in result.stdout
    source = script.read_text()
    assert f"#SBATCH --job-name=mov-hc{epoch}" in source
    assert f"#SBATCH --output=logs/mov-hc{epoch}-%j.out" in source
    assert "#SBATCH --gres=gpu:8" in source and "#SBATCH --nodes=1" in source
    subprocess.run(["bash", "-n", str(script)], check=True)
    Path(env["ARGV_LOG"]).unlink()
    weight.unlink()
    result, argv = run()
    assert result.returncode != 0 and not argv and "no latest/HF fallback" in result.stderr


@pytest.mark.parametrize("epoch", [3, 4])
@pytest.mark.parametrize("phase", ["warm-scores", "score-worker", "summarize", "help"])
def test_isolated_scoring_has_no_av_or_training_imports(tmp_path, epoch, phase):
    contract, reference, assets = prepared_fixture(tmp_path, epoch)
    output = tmp_path / f"history_ep{epoch}"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    protocol.atomic_json(output / "fixture_score.json", reference["scores"]["quicksr"][runner.previous.PARENT_SYSTEM])
    for variant in runner.base.VARIANTS:
        for dimension in protocol.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                 f"{dimension}_eval_results.json", {dimension: [.5, []]})
    program = r'''
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType
script, phase, output, epoch = sys.argv[1:]
output = Path(output)
class NoAV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise ModuleNotFoundError("No module named 'av' (isolated VBench simulation)")
sys.meta_path.insert(0, NoAV())
scope = runpy.run_path(script, run_name="isolated_vbench")["main"].__globals__
assert not any(name.startswith("new_mobile_ov.training") for name in sys.modules)
events = []
scope["base"].core.warm_scores = lambda args: events.append("warm-scores")
scope["base"].verify_generation = lambda output: None
scope["subprocess"].run = lambda argv, **kw: events.append(argv[1])
module = ModuleType("evaluate_vbench_resume")
module.tabulate = lambda folder: json.loads((output / "fixture_score.json").read_text())
sys.modules["evaluate_vbench_resume"] = module
sys.argv = [script, "--epoch", epoch] + (["--help"] if phase == "help" else
           ["--phase", phase, "--output-dir", str(output)])
try:
    scope["main"]()
except SystemExit as error:
    assert phase == "help" and error.code == 0
if phase == "warm-scores":
    assert events == ["warm-scores"]
elif phase == "score-worker":
    assert events and all(name.endswith("evaluate_vbench_resume.py") for name in events)
elif phase == "summarize":
    assert (output / "history_fresh_vs_clean_epoch3.md").is_file()
assert not any(name.startswith("new_mobile_ov.training") for name in sys.modules)
print("ISOLATED_HISTORY_VBENCH_OK")
'''
    result = subprocess.run([sys.executable, "-c", program,
        str(ROOT / "tools/benchmark_mobileov_stage2_history_fresh_epochs_dual.py"), phase, str(output), str(epoch)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATED_HISTORY_VBENCH_OK" in result.stdout
