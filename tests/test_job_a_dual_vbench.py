from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from tools import benchmark_mobileov_job_a_dual as runner
from tools import vbench_joint_native_protocol as protocol
from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option
from test_stage2_media3epochs_dual_vbench import payload as small_payload, fixture_pair
from test_stage2_epoch_vbench import reference_fixture as old_reference

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/vbench_mobileov_job_a_1node8gpu.sbatch"


def payload():
    parent = small_payload()["contract"]
    parent.update(steps=runner.PARENT_STEP, data_summary_sha256="data",
                  frozen_weights_sha256={"smolvlm2": "smol", "vae": "vae"},
                  media_epochs=MediaEpochPlan(984636, 529902, 8, 2).contract())
    value = small_payload()
    value.update(step=runner.FINAL_STEP)
    value["contract"].update(
        steps=runner.FINAL_STEP, scope="first_frame_focus_continuation_public_data_not_exact_UniVideo",
        initialization="stage2_weights_only_fork_fresh_optimizer", init_step=runner.PARENT_STEP,
        init_stage2_sha256=runner.PARENT_SHA256, parent_contract=parent,
        data_summary_sha256="data", frozen_weights_sha256=deepcopy(parent["frozen_weights_sha256"]),
        media_epochs=MediaEpochPlan(984636, 529902, 8, 2, epochs=1, start_epoch=3).contract(),
        first_frame_focus=dict(t2v_first_probability=.5, optimizer_step_origin=runner.PARENT_STEP,
                               history_noise=0, reconstruction_weight=.2, integration_smoke=False),
        ema=False, trainable_components=["connector", "dit"], frozen_components=["smolvlm2", "vae"])
    return value


def reference_fixture(tmp_path):
    root, contract, report, assets = old_reference(tmp_path)
    contract["checkpoint"]["sha256"] = runner.PARENT_SHA256
    contract["training_contract"] = payload()["contract"]["parent_contract"]
    contract["implementation_sha256"][runner.TRAINING_FILE] = protocol.sha256(ROOT / runner.TRAINING_FILE)
    for name, value in (("run_contract.json", contract), ("quicksr_ablation.json", report)):
        protocol.atomic_json(root / name, value)
    return root, contract, report, assets


def current_contract(reference):
    value = deepcopy(reference)
    candidate = payload()
    value.update(system=runner.SYSTEM, training_contract=candidate["contract"],
                 checkpoint=dict(path="job_a.pt", sha256="a" * 64, step=runner.FINAL_STEP, final_step=runner.FINAL_STEP))
    return value


def prepared_fixture(tmp_path):
    root, parent, report, assets = reference_fixture(tmp_path)
    value = current_contract(parent)
    value["job_a_evaluation"] = dict(kind="completed_first_frame_focus_job_a", final_step=runner.FINAL_STEP,
                                    reference_run=str(root), reference_verified=True,
                                    reference_contract_sha256=protocol.fingerprint(parent),
                                    reference_report_sha256=protocol.fingerprint(report))
    return value, report, assets


def test_reader_requires_completed_real_job_a_and_restores_legacy_admission(tmp_path):
    path = tmp_path / "final.pt"
    torch.save(payload(), path)
    original_reader, original_system = runner.base.read_checkpoint, runner.base.SYSTEM
    with pytest.raises(ValueError, match="three-media-epoch"):
        original_reader(path)
    with runner.job_a_policy():
        value, digest, target = runner.base.read_checkpoint(path)
        assert value["step"] == target == 378636 and digest == protocol.sha256(path)
        assert runner.base.SYSTEM == runner.SYSTEM
    assert runner.base.read_checkpoint is original_reader and runner.base.SYSTEM == original_system
    with pytest.raises(ValueError, match="three-media-epoch"):
        original_reader(path)


@pytest.mark.parametrize("change", ["old_parent", "step", "budget", "data", "init", "smoke", "history",
                                    "rec", "first", "lora", "ema", "tasks", "parent", "nan"])
def test_wrong_weights_recipe_partial_run_and_smoke_are_rejected(change):
    value = payload()
    training = value["contract"]
    if change == "old_parent": training["init_stage2_sha256"] = "other"
    elif change == "step": value["step"] = 373977
    elif change == "budget": training["media_epochs"]["steps"] = 10000
    elif change == "data": training["media_epochs"]["videos"] = 83000
    elif change == "init": training["scope"] = "job_b_lora"
    elif change == "smoke": training["first_frame_focus"]["integration_smoke"] = True
    elif change == "history": training["first_frame_focus"]["history_noise"] = .1
    elif change == "rec": training["first_frame_focus"]["reconstruction_weight"] = .1
    elif change == "first": training["first_frame_focus"]["t2v_first_probability"] = .2
    elif change == "lora": training["trainable_components"].append("llm_lora")
    elif change == "ema": training["ema"] = True
    elif change == "tasks": training["tasks"] = ["t2v"]
    elif change == "parent": training["parent_contract"]["steps"] = 130000
    elif change == "nan": value["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.check_job_a_checkpoint(value)


def test_missing_or_changed_checkpoint_never_uses_old_hf_or_parent(tmp_path):
    with pytest.raises(FileNotFoundError, match="no parent/HF fallback"):
        runner.read_checkpoint(tmp_path / "missing.pt")
    path = tmp_path / "final.pt"
    torch.save(payload(), path)
    with pytest.raises(ValueError, match="SHA256"):
        runner.read_checkpoint(path, expected_sha="0" * 64)
    assert runner.read_checkpoint(path, allow_incomplete=True)[0]["step"] == runner.FINAL_STEP
    value = payload()
    value["step"] -= 1
    torch.save(value, path)
    with pytest.raises(ValueError, match="NOT complete"):
        runner.read_checkpoint(path, allow_incomplete=True)


def test_checkpoint_change_during_preflight_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "final.pt"
    torch.save(payload(), path)
    load = torch.load
    def moving_checkpoint(*args, **kwargs):
        value = load(*args, **kwargs)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        return value
    monkeypatch.setattr(torch, "load", moving_checkpoint)
    with pytest.raises(ValueError, match="changed during preflight"):
        runner.read_checkpoint(path)


def test_cli_defaults_and_isolated_smoke_paths():
    args = runner.parse_args([])
    assert args.checkpoint == runner.DEFAULT_CHECKPOINT
    assert "/17780021/" in str(args.checkpoint) and args.output_dir == runner.DEFAULT_OUTPUT
    assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
    assert args.reference_run == runner.REFERENCE
    args = runner.parse_args(["--max-prompts", "2", "--samples-per-prompt", "1"])
    assert args.output_dir.name.endswith("_1x_dual_smoke2")
    args = runner.parse_args(["--checkpoint=/tmp/final.pt", "--output-dir=/tmp/my-evaluation"])
    assert args.checkpoint == Path("/tmp/final.pt") and args.output_dir == Path("/tmp/my-evaluation")
    for flags in (["--allow-incomplete-checkpoint"], ["--samples-per-prompt", "1"], ["--workers", "9"]):
        with pytest.raises(SystemExit):
            runner.parse_args(flags)


def test_cli_works_from_outside_repo_and_advertises_parent_reference(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/benchmark_mobileov_job_a_dual.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--reference-run" in result.stdout and "--expected-sha256" in result.stdout


def test_reference_is_exact_parent_and_legacy_label_survives_job_a_context(tmp_path):
    root, value, report, _ = reference_fixture(tmp_path)
    with runner.job_a_policy():
        assert runner.read_reference(root)[0] == value
        assert runner.base.SYSTEM == runner.SYSTEM
    value["checkpoint"]["sha256"] = "different-parent"
    protocol.atomic_json(root / "run_contract.json", value)
    protocol.atomic_json(root / "quicksr_ablation.json", report)
    with pytest.raises(ValueError, match="exact Job A parent"):
        runner.read_reference(root)


def test_training_script_changes_are_recordable_but_numerical_protocol_is_fixed(tmp_path):
    _, parent, _, _ = reference_fixture(tmp_path)
    value = current_contract(parent)
    runner.compare_protocol(parent, value)
    value["quicksr"]["path"] = "/different/cache/identical.pt"
    runner.compare_protocol(parent, value)
    value["implementation_sha256"]["original"] = "changed"
    with pytest.raises(ValueError, match="Numerical inference"):
        runner.compare_protocol(parent, value)


def test_actual_git_parent_trainer_has_identical_imported_inference_helpers(tmp_path):
    _, parent, _, _ = reference_fixture(tmp_path)
    value = current_contract(parent)
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"c907073:{runner.TRAINING_FILE}"])
    import hashlib
    parent["implementation_sha256"][runner.TRAINING_FILE] = hashlib.sha256(source).hexdigest()
    audit = runner.compare_protocol(parent, value)
    assert audit["policy"] == "changed_training_code_but_identical_imported_inference_helpers"
    assert set(audit["helpers"]) == {"load_stack", "autocast", "encode_anchor"}


def test_changed_imported_inference_function_is_not_hidden_as_a_training_edit(tmp_path, monkeypatch):
    _, parent, _, _ = reference_fixture(tmp_path)
    value = current_contract(parent)
    parent["implementation_sha256"][runner.TRAINING_FILE] = "other-source"
    source = (ROOT / runner.TRAINING_FILE).read_text().replace("def load_stack(args, device):", "def load_stack(args, device=None):")
    assert source != (ROOT / runner.TRAINING_FILE).read_text()
    monkeypatch.setattr(runner, "historical_training_source", lambda _: (source, "old-commit"))
    with pytest.raises(ValueError, match="imported inference helpers differ"):
        runner.compare_protocol(parent, value)


@pytest.mark.parametrize("field", ["seed", "guidance", "video_steps_per_stage", "prompts", "library_versions",
                                    "external_anchor", "exports", "connector_spec", "vbench_source_commit"])
def test_inference_differences_from_parent_fail_before_generation(tmp_path, field):
    _, parent, _, _ = reference_fixture(tmp_path)
    value = current_contract(parent)
    value[field] = "changed"
    with pytest.raises(ValueError, match="Evaluation differs"):
        runner.compare_protocol(parent, value)


@pytest.mark.parametrize("field", ["data_summary_sha256", "processor_sha256", "frozen_weights_sha256"])
def test_frozen_stack_and_data_must_still_match_parent(tmp_path, field):
    _, parent, _, _ = reference_fixture(tmp_path)
    value = current_contract(parent)
    value["training_contract"][field] = "changed"
    with pytest.raises(ValueError, match="data/frozen"):
        runner.compare_protocol(parent, value)


def test_preflight_records_correct_identity_and_does_not_modify_parent(tmp_path, monkeypatch):
    root, parent, _, _ = reference_fixture(tmp_path)
    previous = protocol.sha256(root / "run_contract.json")
    current = current_contract(parent)
    def prepare(args):
        assert runner.base.SYSTEM == runner.SYSTEM and runner.base.read_checkpoint is runner.read_checkpoint
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(current))))
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=True))
        return deepcopy(current)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "job-a", reference_run=root, max_prompts=0)
    contract = runner.prepare(args)
    assert runner.prepare(args) == contract
    assert contract["system"] == runner.SYSTEM and contract["training_complete"] is True
    assert contract["checkpoint"]["step"] == 378636
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert contract["job_a_evaluation"]["reference_verified"] is True
    assert contract["job_a_evaluation"]["additional_updates"] == 94659
    assert protocol.sha256(root / "run_contract.json") == previous
    with pytest.raises(ValueError, match="separate Job A"):
        runner.prepare(SimpleNamespace(**{**vars(args), "output_dir":root}))


def test_actual_dual_mp4_generation_reuses_exact_inference_and_cached_pairs(tmp_path, monkeypatch):
    from tools import infer_mobileov_stage1 as infer
    from new_mobile_ov.generation import quicksrnet

    frames, kwargs = fixture_pair(tmp_path)
    contract = kwargs["contract"]
    path = tmp_path / "weights.pt"
    torch.save(payload(), path)
    contract.update(system=runner.SYSTEM, training_complete=True,
                    checkpoint=dict(path=str(path), sha256=protocol.sha256(path), step=runner.FINAL_STEP),
                    job_a_evaluation=dict(final_step=runner.FINAL_STEP))
    protocol.atomic_json(tmp_path / "run_contract.json", contract)
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-export-integration-test")
    calls = []
    def encoder(prompt, **kwargs):
        assert "image" not in kwargs
        return torch.zeros(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 4)
    def load_models(args, loaded, device):
        assert loaded["step"] == runner.FINAL_STEP
        calls.append("load")
        return encoder, lambda *parts:parts, None, None, None, dict(trained_dit_loaded=True)
    def generate(*args, **kwargs):
        calls.append("generate")
        assert "anchor_latent" not in kwargs and kwargs["num_frames"] == 49
        assert (kwargs["first_steps"], kwargs["video_steps"], kwargs["guidance"], kwargs["video_guidance"]) == (20,10,7,5)
        return frames, [dict(unit=0, generated=True)]
    monkeypatch.setattr(runner.base.stage2, "load_models", load_models)
    monkeypatch.setattr(infer, "autocast", lambda _:nullcontext())
    monkeypatch.setattr(infer, "generate_frames", generate)
    monkeypatch.setattr(quicksrnet, "load_quicksrnet", lambda **kwargs:(torch.nn.Upsample(scale_factor=2), None))
    args = SimpleNamespace(output_dir=tmp_path, rank=0, workers=1)
    # The wrapper pins a CPU device only inside this mechanical export test.
    original = runner.base.generation_worker
    monkeypatch.setattr(runner.base, "generation_worker", lambda value:original(value, device=torch.device("cpu")))
    runner.generation_worker(args)
    runner.generation_worker(args)
    assert calls == ["load", "generate"]
    with runner.job_a_policy():
        runner.base.verify_generation(tmp_path)
    for variant in runner.base.VARIANTS:
        record = protocol.read_json(tmp_path / variant / "video_metadata/0000_0.json")
        assert record["system"] == runner.SYSTEM and record["first_frame_generated"] is True
        assert record["external_anchor"] is False
    assert protocol.read_json(tmp_path / "generation_summary.json")["checkpoint"]["path"] == str(path)


def test_assets_and_parent_mutation_are_detected(tmp_path):
    contract, _, assets = prepared_fixture(tmp_path)
    protocol.atomic_json(tmp_path / "score_assets.json", [dict(assets[0], path="relocated.pt")])
    runner.verify_score_assets(tmp_path, contract)
    protocol.atomic_json(tmp_path / "score_assets.json", [dict(assets[0], sha256="different")])
    with pytest.raises(ValueError, match="evaluation weights"):
        runner.verify_score_assets(tmp_path, contract)
    contract["job_a_evaluation"]["reference_contract_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed after"):
        runner.verified_reference(contract)


def test_full_summary_has_all_dimensions_both_variants_and_parent_deltas(tmp_path, monkeypatch):
    contract, parent, assets = prepared_fixture(tmp_path)
    output = tmp_path / "job-a"
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output / "score_assets.json", assets)
    score = deepcopy(parent["scores"]["quicksr"][runner.PARENT_SYSTEM])
    score["total_score"] += .01
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = lambda _:deepcopy(score)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _:None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:None)
    for variant in runner.base.VARIANTS:
        for dimension in runner.base.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json",
                                 {dimension:[.5, []]})
    runner.summarize(SimpleNamespace(output_dir=output))
    report = protocol.read_json(output / "quicksr_ablation.json")
    assert set(report["scores"]["quicksr"]) == {runner.SYSTEM}
    comparison = protocol.read_json(output / "job_a_vs_epoch3.json")
    assert len(comparison["rows"]) == 19
    assert comparison["rows"]["total_score"]["quicksr"]["difference_pp"] == pytest.approx(1.)
    assert "not an isolated loss ablation" in (output / "job_a_vs_epoch3.md").read_text()
    contract["max_prompts"] = 2
    protocol.atomic_json(output / "run_contract.json", contract)
    with pytest.raises(ValueError, match="No full Job A"):
        runner.summarize(SimpleNamespace(output_dir=output))


def test_orchestration_checks_assets_before_generation_and_scores_all_variants(tmp_path, monkeypatch):
    contract, _, _ = prepared_fixture(tmp_path)
    args = SimpleNamespace(phase="run", output_dir=tmp_path / "job-a", workers=8, max_prompts=0,
                           vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda:args)
    monkeypatch.setattr(runner, "prepare", lambda _:contract)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _:list(map(str, range(8))))
    events = []
    monkeypatch.setattr(runner, "verify_score_assets", lambda *args:events.append("assets"))
    monkeypatch.setattr(runner.base, "verify_generation", lambda _:events.append("verify_generation"))
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs:events.append(option(argv, "--phase")))
    def workers(commands, devices, folder, **kwargs):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        assert all(argv[1].endswith("benchmark_mobileov_job_a_dual.py") for argv in commands)
        events.append(option(commands[0], "--phase"))
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert events == ["warm-scores", "assets", "generation-worker", "verify_generation", "score-worker", "summarize"]


def test_launcher_uses_job_a_final_and_read_only_parent_reference(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, env, run = launch
    path = root / "output/mobileov_stage2_first_frame_focus/17780021/stage2_dit_connector_latest.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"job a final")
    reference = root / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(reference / name, {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
(output/"job_a_vs_epoch3.json").write_text("{}")
(output/"job_a_vs_epoch3.md").write_text("mock Job A comparison")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert "tools/benchmark_mobileov_job_a_dual.py" in argv
    assert option(argv, "--checkpoint") == str(path.relative_to(root))
    assert option(argv, "--reference-run") == str(reference.relative_to(root))
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--seed") == "2026" and "--allow-incomplete-checkpoint" not in argv
    assert "mock Job A comparison" in result.stdout
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        assert protocol.read_json(reference / name) == {}
    Path(env["ARGV_LOG"]).unlink()
    path.unlink()
    result, argv = run()
    assert result.returncode != 0 and not argv and "no parent/HF fallback" in result.stderr
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "#SBATCH --gres=gpu:8" in SCRIPT.read_text()


@pytest.mark.parametrize("overrides", [{"SAMPLES_PER_PROMPT":"1"}, {"MAX_PROMPTS":"-1"}, {"SAMPLES_PER_PROMPT":"6"}])
def test_invalid_launcher_settings_fail_before_environment_setup(launch, monkeypatch, overrides):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    _, env, run = launch
    result, argv = run(**overrides)
    assert result.returncode != 0 and not argv and not Path(env["SETUP_LOG"]).exists()
