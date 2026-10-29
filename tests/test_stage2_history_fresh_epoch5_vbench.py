from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import benchmark_mobileov_stage2_history_fresh_epoch5_dual as runner
import test_stage2_history_fresh_epochs_vbench as helpers
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

protocol, ROOT = helpers.protocol, helpers.ROOT
SCRIPT = ROOT / "scripts/vbench_mobileov_stage2_history_fresh_epoch5_1node8gpu.sbatch"


def payload():
    return dict(helpers.payload(4), step=473295)


def prepared_fixture(tmp_path):
    result, reference, assets = helpers.prepared_fixture(tmp_path, 4)
    result.update(system=runner.SYSTEM, training_complete=True)
    result["checkpoint"].update(path=str(runner.DEFAULT_CHECKPOINT), step=473295)
    result["history_fresh_evaluation"].update(epoch=5, expected_step=473295)
    return result, reference, assets


def test_cli_is_explicitly_history_epoch5_not_clean_resume_or_moving_latest():
    args = runner.parse_args([])
    assert args.checkpoint == runner.DEFAULT_CHECKPOINT
    assert "17892009" in str(args.checkpoint) and args.checkpoint.name == "stage2_dit_connector_step473295.pt"
    assert args.output_dir == runner.DEFAULT_OUTPUT and "history_fresh_epoch5" in args.output_dir.name
    assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
    for flags in (["--samples-per-prompt", "1"], ["--allow-incomplete-checkpoint"], ["--workers", "16"]):
        with pytest.raises(SystemExit):
            runner.parse_args(flags)
    smoke = runner.parse_args(["--max-prompts", "2", "--samples-per-prompt", "1"])
    assert smoke.output_dir.name.endswith("_1x_dual_smoke2")


def test_final_reader_checks_real_step_and_does_not_rewrite_training_contract(tmp_path):
    value = payload()
    original = deepcopy(value["contract"])
    path = tmp_path / "archive.pt"
    torch.save(value, path)
    before = runner.base.check_media_checkpoint, runner.base.SYSTEM
    with runner.epoch5_policy():
        actual, digest, target = runner.base.read_checkpoint(path)
        assert actual["step"] == target == 473295 and digest == protocol.sha256(path)
        assert runner.base.SYSTEM == runner.SYSTEM
        with pytest.raises(ValueError, match="SHA256 changed"):
            runner.base.read_checkpoint(path, expected_sha="0" * 64)
    assert original == value["contract"] == actual["contract"]
    assert before == (runner.base.check_media_checkpoint, runner.base.SYSTEM)
    assert runner.clean.EPOCH == 4 and 5 not in runner.history.EPOCH_STEPS


@pytest.mark.parametrize("change", ["step", "clean_resume", "fork", "sigma", "optimizer", "lora", "rec", "lr", "nan"])
def test_wrong_checkpoints_never_pass_even_for_smoke(change):
    value = payload()
    if change == "step": value["step"] = 378636
    elif change == "clean_resume": value["contract"] = helpers.clean.payload()["contract"]
    elif change == "fork": value["contract"]["initialization"] = "stage2_weights_only_fork_fresh_optimizer"
    elif change == "sigma": value["contract"]["history_corruption"]["max_sigma"] = .2
    elif change == "optimizer": value["optimizer"] = {}
    elif change == "lora": value["smolvlm2_lora"] = {}
    elif change == "rec": value["contract"]["auxiliary_reconstruction"]["weight"] = .2
    elif change == "lr": value["contract"]["lr"] = 1e-4
    else: value["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.check_checkpoint(value, allow_incomplete=True)


def test_prepare_marks_final_completion_and_preserves_reference(tmp_path, monkeypatch):
    result, _, _ = prepared_fixture(tmp_path)
    root = Path(result["history_fresh_evaluation"]["reference_run"])
    before = protocol.sha256(root / "run_contract.json")
    def prepare(args):
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=True))
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(result))))
        return deepcopy(result)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "hc5", reference_run=root, max_prompts=0)
    contract = runner.prepare(args)
    assert runner.prepare(args) == contract
    assert contract["training_complete"] is True and contract["history_fresh_evaluation"]["epoch"] == 5
    assert contract["history_fresh_evaluation"]["training_run"] == "17892009"
    assert "parent_contract" not in contract["training_contract"]
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert protocol.sha256(root / "run_contract.json") == before
    assert runner.checked_contract(args.output_dir) == contract
    contract["training_complete"] = False
    protocol.atomic_json(args.output_dir / "run_contract.json", contract)
    with pytest.raises(ValueError, match="completed fresh-history"):
        runner.checked_contract(args.output_dir)


def test_summary_keeps_epoch5_identity_and_all_raw_quicksr_dimensions(tmp_path, monkeypatch):
    contract, reference, assets = prepared_fixture(tmp_path)
    output = tmp_path / "hc5"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = lambda folder: deepcopy(reference["scores"][
        "no_quicksr" if "no_quicksr" in folder.parts else "quicksr"][runner.history.previous.PARENT_SYSTEM])
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
    assert set(report["scores"]["quicksr"]) == {"stage2_history_fresh_epoch5"}
    comparison = protocol.read_json(output / "history_fresh_vs_clean_epoch3.json")
    assert len(comparison["rows"]) == 19
    assert comparison["rows"]["total_score"]["quicksr"]["difference_pp"] == 0
    assert "History Ep5 QuickSR" in (output / "history_fresh_vs_clean_epoch3.md").read_text()


def test_orchestration_uses_epoch5_entrypoint_for_all_workers(tmp_path, monkeypatch):
    args = SimpleNamespace(phase="run", output_dir=tmp_path / "hc5", workers=8,
                           max_prompts=0, vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare", lambda _: {})
    monkeypatch.setattr(runner.history, "verify_score_assets", lambda *a: None)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    calls = []
    def process(argv, **kwargs):
        assert argv[1].endswith("history_fresh_epoch5_dual.py")
        calls.append(option(argv, "--phase"))
    def workers(commands, devices, folder, **kwargs):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        assert all(argv[1].endswith("history_fresh_epoch5_dual.py") for argv in commands)
        calls.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert calls == ["warm-scores", "generation-worker", "score-worker", "summarize"]


def test_slurm_job_has_distinct_name_final_archive_and_dual_outputs(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, env, run = launch
    weight = root / runner.DEFAULT_CHECKPOINT.relative_to(ROOT)
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fresh-history final weights")
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
    (output/(name+".md")).write_text("mock hc5 report")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--checkpoint") == str(weight.relative_to(root))
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--output-dir").endswith("history_fresh_epoch5_step473295_5x_dual")
    assert "mock hc5 report" in result.stdout
    assert "#SBATCH --job-name=mov-hc5" in SCRIPT.read_text()
    assert "#SBATCH --output=logs/mov-hc5-%j.out" in SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:8" in SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    Path(env["ARGV_LOG"]).unlink()
    weight.unlink()
    result, argv = run()
    assert result.returncode != 0 and not argv and "no latest/HF fallback" in result.stderr


@pytest.mark.parametrize("phase", ["warm-scores", "score-worker", "summarize", "help"])
def test_scoring_starts_without_av_or_training_imports(tmp_path, phase):
    contract, reference, assets = prepared_fixture(tmp_path)
    output = tmp_path / "hc5"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    protocol.atomic_json(output / "fixture_score.json", reference["scores"]["quicksr"][runner.history.previous.PARENT_SYSTEM])
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
script, phase, output = sys.argv[1:]
output = Path(output)
class NoAV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise ModuleNotFoundError("No module named 'av' (isolated VBench simulation)")
sys.meta_path.insert(0, NoAV())
scope = runpy.run_path(script, run_name="isolated_vbench")["main"].__globals__
events = []
scope["base"].core.warm_scores = lambda args: events.append("warm-scores")
scope["base"].verify_generation = lambda output: None
scope["subprocess"].run = lambda argv, **kw: events.append(argv[1])
module = ModuleType("evaluate_vbench_resume")
module.tabulate = lambda folder: json.loads((output / "fixture_score.json").read_text())
sys.modules["evaluate_vbench_resume"] = module
sys.argv = [script] + (["--help"] if phase == "help" else ["--phase", phase, "--output-dir", str(output)])
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
print("HC5_ISOLATED_VBENCH_OK")
'''
    result = subprocess.run([sys.executable, "-c", program,
        str(ROOT / "tools/benchmark_mobileov_stage2_history_fresh_epoch5_dual.py"), phase, str(output)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HC5_ISOLATED_VBENCH_OK" in result.stdout


def test_sources_used_by_existing_epoch3_4_jobs_remain_byte_identical():
    for name in ("tools/benchmark_mobileov_stage2_history_fresh_epochs_dual.py",
                 "tools/benchmark_mobileov_stage2_resume_epoch4_dual.py",
                 "scripts/run_vbench_mobileov_history_fresh_epoch.sh"):
        before = subprocess.check_output(["git", "-C", str(ROOT), "show", f"79d5a88:{name}"])
        assert (ROOT / name).read_bytes() == before
