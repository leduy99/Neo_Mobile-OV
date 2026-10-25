from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import benchmark_mobileov_stage2_resume_epoch4_dual as runner
from tools import vbench_joint_native_protocol as protocol
from new_mobile_ov.training.stage2_alignment import reconstruction_contract
from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan, METRIC_TASKS
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option
from test_job_a_dual_vbench import reference_fixture as old_reference, payload as job_a_payload
from test_stage2_media3epochs_dual_vbench import payload as small_payload

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/vbench_mobileov_stage2_resume_epoch4_1node8gpu.sbatch"


def original_training():
    result = small_payload()["contract"]
    result.update(format=runner.base.stage2.FORMAT,
        initialization="stage1_connector_and_released_multistep_dit", init_step=15000,
        scope="fresh_multimodal_stage2; public_data; not_exact_UniVideo_recipe",
        trainable_components=["connector", "dit"], frozen_components=["smolvlm2", "vae"],
        frozen_weights_sha256={"smolvlm2": "smol", "vae": "vae"}, data_summary_sha256="data",
        auxiliary_reconstruction=reconstruction_contract(.1),
        video_units="T2V/reference:0_to6; anchor:1_to6; teacher_forced_causal_history",
        tasks=list(METRIC_TASKS), lr=2e-5, warmup=50, ema=False,
        world_size=8, accumulation=2, steps=283977,
        media_epochs=MediaEpochPlan(984636, 529902, 8, 2).contract())
    return result


def payload():
    result = small_payload()
    result.update(step=runner.EPOCH_STEP, contract=original_training())
    result["contract"].update(world_size=16, accumulation=1, steps=runner.TARGET_STEP,
        media_epochs=MediaEpochPlan(984636, 529902, 16, 1, epochs=5,
                                    allow_partial_cycle=True, rng_world_size=8).contract())
    return result


def reference_fixture(tmp_path):
    root, reference, report, assets = old_reference(tmp_path)
    reference["training_contract"] = original_training()
    report["protocol"] = reference
    protocol.atomic_json(root / "run_contract.json", reference)
    protocol.atomic_json(root / "quicksr_ablation.json", report)
    return root, reference, report, assets


def current_contract(reference):
    result = deepcopy(reference)
    result.update(system=runner.SYSTEM, training_complete=runner.EPOCH_STEP == runner.TARGET_STEP,
        training_contract=payload()["contract"],
        checkpoint=dict(path=str(runner.DEFAULT_CHECKPOINT), sha256="a" * 64,
                        step=runner.EPOCH_STEP, final_step=runner.TARGET_STEP))
    return result


def prepared_fixture(tmp_path):
    root, reference, report, assets = reference_fixture(tmp_path)
    result = current_contract(reference)
    result[f"epoch{runner.EPOCH}_evaluation"] = dict(kind=f"completed_epoch{runner.EPOCH}_of_five_epoch_resume",
        epoch=runner.EPOCH, expected_step=runner.EPOCH_STEP, planned_final_step=runner.TARGET_STEP,
        reference_run=str(root), reference_verified=True,
        reference_contract_sha256=protocol.fingerprint(reference),
        reference_report_sha256=protocol.fingerprint(report))
    return result, report, assets


def test_reader_pins_epoch4_without_weakening_legacy_or_claiming_final_completion(tmp_path):
    path = tmp_path / "epoch4.pt"
    torch.save(payload(), path)
    old_check, old_system = runner.base.check_media_checkpoint, runner.base.SYSTEM
    with pytest.raises(ValueError, match="three-media-epoch"):
        runner.base.read_checkpoint(path)
    with runner.epoch4_policy():
        result, digest, target = runner.base.read_checkpoint(path)
        assert result["step"] == 378636 and target == 473295
        assert digest == protocol.sha256(path) and runner.base.SYSTEM == runner.SYSTEM
        with pytest.raises(ValueError, match="SHA256 changed"):
            runner.base.read_checkpoint(path, expected_sha="0" * 64)
    assert runner.base.check_media_checkpoint is old_check and runner.base.SYSTEM == old_system


@pytest.mark.parametrize("change", ["step", "job_a", "resume", "history", "lora", "lr", "reconstruction",
                                    "batch", "rng", "data_count", "budget", "tasks", "nan"])
def test_wrong_checkpoint_or_recipe_is_rejected_even_for_smoke(change):
    result = payload()
    training = result["contract"]
    if change == "step": result["step"] = 410000
    elif change == "job_a": result = job_a_payload()
    elif change == "resume": result["optimizer"] = {}
    elif change == "history": training["history_corruption"] = {}
    elif change == "lora": result["smolvlm2_lora"] = {}
    elif change == "lr": training["lr"] = 1e-4
    elif change == "reconstruction": training["auxiliary_reconstruction"]["weight"] = .2
    elif change == "batch": training["accumulation"] = 2
    elif change == "rng": training["media_epochs"]["rng_layout"]["world_size"] = 16
    elif change == "data_count": training["media_epochs"]["videos"] = 83000
    elif change == "budget": training["steps"] = 378636
    elif change == "tasks": training["tasks"] = ["t2v"]
    else: result["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.check_checkpoint(result, allow_incomplete=True)


def test_cli_uses_immutable_known_archive_eight_workers_and_both_versions():
    args = runner.parse_args([])
    assert args.checkpoint == runner.DEFAULT_CHECKPOINT
    assert args.checkpoint.name == "stage2_dit_connector_step378636.pt" and "17892008" in str(args.checkpoint)
    assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
    assert args.output_dir == runner.DEFAULT_OUTPUT and not args.allow_incomplete_checkpoint
    for flags in (["--samples-per-prompt", "1"], ["--allow-incomplete-checkpoint"], ["--workers", "16"]):
        with pytest.raises(SystemExit):
            runner.parse_args(flags)
    smoke = runner.parse_args(["--max-prompts", "2", "--samples-per-prompt", "1"])
    assert smoke.output_dir.name.endswith("_1x_dual_v2_smoke2")


def test_epoch4_protocol_accepts_only_layout_and_total_epoch_changes(tmp_path):
    _, reference, _, _ = reference_fixture(tmp_path)
    result = current_contract(reference)
    audit = runner.compare_protocol(reference, result)
    assert audit["policy"] == "identical_trainer_source"
    assert "parent_contract" not in result["training_contract"]
    for name in ("seed", "exports", "guidance", "library_versions", "prompts", "implementation_sha256"):
        wrong = deepcopy(result)
        wrong[name] = "changed"
        with pytest.raises((ValueError, TypeError, AttributeError)):
            runner.compare_protocol(reference, wrong)
    for name in ("data_summary_sha256", "init_step", "frozen_weights_sha256"):
        wrong = deepcopy(result)
        wrong["training_contract"][name] = "changed"
        with pytest.raises(ValueError, match="changed the training recipe"):
            runner.compare_protocol(reference, wrong)


def test_changed_training_code_requires_identical_imported_inference_helpers(tmp_path):
    import hashlib
    _, reference, _, _ = reference_fixture(tmp_path)
    result = current_contract(reference)
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"c907073:{runner.previous.TRAINING_FILE}"])
    reference["implementation_sha256"][runner.previous.TRAINING_FILE] = hashlib.sha256(source).hexdigest()
    audit = runner.compare_protocol(reference, result)
    assert audit["policy"] == "changed_training_code_but_identical_imported_inference_helpers"


def test_prepare_preserves_parent_and_records_epoch4_as_intermediate_of_five(tmp_path, monkeypatch):
    root, reference, _, _ = reference_fixture(tmp_path)
    before = protocol.sha256(root / "run_contract.json")
    result = current_contract(reference)
    def prepare(args):
        assert runner.base.SYSTEM == runner.SYSTEM
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(result))))
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=False))
        return deepcopy(result)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "epoch4", reference_run=root, max_prompts=0)
    actual = runner.prepare(args)
    assert runner.prepare(args) == actual
    assert actual["training_complete"] is False and actual["epoch4_evaluation"]["epoch"] == 4
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert protocol.sha256(root / "run_contract.json") == before
    assert "0.2" not in actual["scope"] and "history corruption" in actual["scope"]
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(SimpleNamespace(**dict(vars(args), output_dir=root)))


def test_generation_worker_reuses_existing_numerical_path_and_checkpoint_hash_guard(tmp_path, monkeypatch):
    contract, _, _ = prepared_fixture(tmp_path)
    protocol.atomic_json(tmp_path / "run_contract.json", contract)
    path = tmp_path / "weights.pt"
    torch.save(payload(), path)
    calls = []
    def worker(args, *, device=None):
        value, _, target = runner.base.read_checkpoint(path)
        calls.append((value["step"], target, runner.base.SYSTEM))
    monkeypatch.setattr(runner.base, "generation_worker", worker)
    runner.generation_worker(SimpleNamespace(output_dir=tmp_path))
    assert calls == [(378636, 473295, runner.SYSTEM)]
    assert runner.base.SYSTEM == runner.previous.PARENT_SYSTEM


def test_summary_has_all_dimensions_and_keeps_incomplete_five_epoch_status(tmp_path, monkeypatch):
    contract, parent_report, assets = prepared_fixture(tmp_path)
    output = tmp_path / "epoch4"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    score = deepcopy(parent_report["scores"]["quicksr"][runner.previous.PARENT_SYSTEM])
    score["total_score"] += .02
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = lambda _: deepcopy(score)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    for variant in runner.base.VARIANTS:
        for dimension in protocol.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                 f"{dimension}_eval_results.json", {dimension: [.5, []]})
    runner.summarize(SimpleNamespace(output_dir=output))
    report = protocol.read_json(output / "quicksr_ablation.json")
    assert report["status"] == "complete" and report["protocol"]["training_complete"] is False
    result = protocol.read_json(output / "epoch4_vs_epoch3.json")
    assert len(result["rows"]) == 19
    assert result["rows"]["total_score"]["quicksr"]["difference_pp"] == pytest.approx(2.)
    assert "Epoch 4 QuickSR" in (output / "epoch4_vs_epoch3.md").read_text()
    protocol.atomic_json(output / "score_assets.json", [dict(assets[0], sha256="changed")])
    with pytest.raises(ValueError, match="evaluation weights"):
        runner.verify_score_assets(output, contract)
    contract["epoch4_evaluation"]["reference_report_sha256"] = "changed"
    with pytest.raises(ValueError, match="reference changed"):
        runner.verified_reference(contract)


def test_full_orchestration_runs_eight_workers_for_generation_and_scoring(tmp_path, monkeypatch):
    contract, _, _ = prepared_fixture(tmp_path)
    args = SimpleNamespace(epoch=4, phase="run", output_dir=tmp_path / "epoch4", workers=8, max_prompts=0,
                           vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare", lambda _: contract)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    events = []
    monkeypatch.setattr(runner, "verify_score_assets", lambda *a: events.append("assets"))
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: events.append("verify_generation"))
    def process(argv, **kwargs):
        events.append(option(argv, "--phase"))
    def workers(commands, devices, folder, **kwargs):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        assert all(argv[1].endswith("benchmark_mobileov_stage2_resume_epoch4_dual.py") for argv in commands)
        events.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert events == ["warm-scores", "assets", "generation-worker", "verify_generation", "score-worker", "summarize"]


def test_berzelius_job_pins_run17892008_and_evaluates_both_versions(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, env, run = launch
    weight = root / "output/mobileov_stage2_resume5epochs16gpu/17892008/stage2_dit_connector_step378636.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"archived epoch4 weights")
    reference = root / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(reference / name, {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
for name in ("quicksr_ablation", "epoch4_vs_epoch3"):
    (output/(name+".json")).write_text("{}")
    (output/(name+".md")).write_text("mock epoch4 report")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert "tools/benchmark_mobileov_stage2_resume_epoch4_dual.py" in argv
    assert option(argv, "--checkpoint") == str(weight.relative_to(root))
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--seed") == "2026" and "--allow-incomplete-checkpoint" not in argv
    assert option(argv, "--epoch") == "4"
    assert option(argv, "--output-dir").endswith("resume_epoch4_step378636_5x_dual_v2")
    assert "mock epoch4 report" in result.stdout
    assert "#SBATCH --gres=gpu:8" in SCRIPT.read_text()
    assert "#SBATCH --nodes=1" in SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    Path(env["ARGV_LOG"]).unlink()
    weight.unlink()
    result, argv = run()
    assert result.returncode != 0 and not argv and "no latest/HF fallback" in result.stderr


@pytest.mark.parametrize("phase", ["warm-scores", "score-worker", "summarize", "help"])
@pytest.mark.parametrize("epoch", [4, 5])
def test_scoring_processes_start_without_av_or_training_dependencies(tmp_path, phase, epoch):
    with runner.evaluation_epoch(epoch):
        contract, parent_report, assets = prepared_fixture(tmp_path)
    output = tmp_path / f"epoch{epoch}"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    score = parent_report["scores"]["quicksr"][runner.previous.PARENT_SYSTEM]
    protocol.atomic_json(output / "fixture_score.json", score)
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
            raise ModuleNotFoundError("No module named 'av' (simulated isolated VBench environment)")
sys.meta_path.insert(0, NoAV())
namespace = runpy.run_path(script, run_name="isolated_vbench")
assert not any(name == "new_mobile_ov.training" or name.startswith("new_mobile_ov.training.")
               for name in sys.modules), "Scoring imported the training stack"
scope = namespace["main"].__globals__
events = []
scope["base"].core.warm_scores = lambda args: events.append("warm-scores")
scope["base"].verify_generation = lambda folder: events.append("verify-generation")
def subprocess_run(argv, **kwargs):
    events.append(argv[1])
scope["subprocess"].run = subprocess_run
tabulator = ModuleType("evaluate_vbench_resume")
tabulator.tabulate = lambda folder: json.loads((output / "fixture_score.json").read_text())
sys.modules["evaluate_vbench_resume"] = tabulator
sys.argv = ([script, "--epoch", epoch, "--help"] if phase == "help" else
            [script, "--epoch", epoch, "--phase", phase, "--output-dir", str(output)])
try:
    scope["main"]()
except SystemExit as error:
    assert phase == "help" and error.code == 0
if phase == "warm-scores":
    assert events == ["warm-scores"]
elif phase == "score-worker":
    assert events and all(name.endswith("evaluate_vbench_resume.py") for name in events)
elif phase == "summarize":
    assert (output / f"epoch{epoch}_vs_epoch3.md").is_file()
assert not any(name == "new_mobile_ov.training" or name.startswith("new_mobile_ov.training.")
               for name in sys.modules), "Scoring imported the training stack"
print("ISOLATED_VBENCH_OK", phase)
'''
    result = subprocess.run([sys.executable, "-c", program,
                             str(ROOT / "tools/benchmark_mobileov_stage2_resume_epoch4_dual.py"),
                             phase, str(output), str(epoch)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ISOLATED_VBENCH_OK {phase}" in result.stdout
