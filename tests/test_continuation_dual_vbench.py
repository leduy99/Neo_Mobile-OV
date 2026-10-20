from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from tools import benchmark_mobileov_continuations_dual as runner
from tools import vbench_joint_native_protocol as protocol
from new_mobile_ov.training import job_b_model as job
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from new_mobile_ov.training.stage2_history import history_contract, MAX_HISTORY_SIGMA
from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from test_job_a_dual_vbench import payload as job_a_payload, reference_fixture
from test_stage2_media3epochs_dual_vbench import fixture_pair
from test_mobileov_job_b import Backbone, Processor
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

ROOT = Path(__file__).resolve().parents[1]


def payload(experiment):
    value = job_a_payload()
    training = value["contract"]
    training["format"] = runner.base.stage2.FORMAT
    training.pop("first_frame_focus")
    if experiment == "uniform_history":
        training.update(scope="uniform_history_continuation_public_data_not_exact_UniVideo",
                        history_corruption=history_contract(MAX_HISTORY_SIGMA),
                        auxiliary_reconstruction=dict(weight=.1),
                        video_units="T2V/reference:uniform0_to6; anchor:uniform1_to6; teacher_forced_corrupted_history")
    else:
        parent = training["parent_contract"]
        training = {key: training[key] for key in ("config_sha256", "processor", "processor_sha256", "max_tokens",
                    "data_summary_sha256", "frozen_weights_sha256", "parent_contract", "media_epochs",
                    "init_step", "init_stage2_sha256", "steps", "ema")}
        spec = dict(rank=16, alpha=32, dropout=0, targets=list(job.TARGETS))
        training.update(format=job.FORMAT, integration_smoke=False, lora_spec=spec,
                        trainable_components=["LLM_LoRA", "MCP", "DiT"], reconstruction_weight=.2,
                        t2v_first_probability=.5, history_noise=0, kl=False, qa_every=8, qa_weight=.1)
        value.update(format=job.FORMAT, contract=training, lora_spec=spec,
                     smolvlm2_lora={"projection.lora_A": torch.ones(16, 32), "projection.lora_B": torch.ones(32, 16)})
        assert "initial_weights_sha256" not in training and "smolvlm2_sha256" not in training
        assert parent["max_tokens"] == training["max_tokens"]
    return value


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
def test_completed_reader_pins_recipe_and_restores_legacy_reader(tmp_path, experiment):
    path = tmp_path / "weights.pt"
    torch.save(payload(experiment), path)
    old = runner.base.read_checkpoint, runner.base.SYSTEM, runner.base.stage2.load_models
    with runner.continuation_policy(experiment):
        value, digest, target = runner.base.read_checkpoint(path)
        assert value["step"] == target == 378636 and digest == protocol.sha256(path)
        assert runner.base.SYSTEM == runner.SYSTEMS[experiment]
        if experiment == "job_b":
            assert runner.base.stage2.load_models is runner.load_job_b_models
    assert old == (runner.base.read_checkpoint, runner.base.SYSTEM, runner.base.stage2.load_models)
    other = "job_b" if experiment == "uniform_history" else "uniform_history"
    with pytest.raises(ValueError): runner.check_checkpoint(value, other)
    with pytest.raises(ValueError, match="SHA256"):
        runner.read_checkpoint(path, experiment=experiment, expected_sha="0" * 64)
    with pytest.raises(FileNotFoundError, match="no parent/HF fallback"):
        runner.read_checkpoint(tmp_path / "missing.pt", experiment=experiment)


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
@pytest.mark.parametrize("change", ["step", "parent", "budget", "frozen", "nan", "recipe"])
def test_wrong_partial_or_nonfinite_checkpoints_are_rejected(experiment, change):
    value = payload(experiment)
    if change == "step": value["step"] -= 1
    elif change == "parent": value["contract"]["init_stage2_sha256"] = "wrong"
    elif change == "budget": value["contract"]["media_epochs"]["videos"] = 83000
    elif change == "frozen": value["contract"]["frozen_weights_sha256"] = {"smolvlm2": "wrong"}
    elif change == "nan": value["dit"]["weight"][0, 0] = float("nan")
    elif experiment == "job_b": value["contract"]["reconstruction_weight"] = .1
    else: value["contract"]["history_corruption"]["max_sigma"] = 0
    with pytest.raises(ValueError): runner.check_checkpoint(value, experiment)


@pytest.mark.parametrize("change", ["missing", "zero", "nan", "dtype", "unpaired", "spec"])
def test_lora_cannot_be_missing_untrained_or_incompatible(change):
    value = payload("job_b")
    if change == "missing": value.pop("smolvlm2_lora")
    elif change == "zero": value["smolvlm2_lora"]["projection.lora_B"].zero_()
    elif change == "nan": value["smolvlm2_lora"]["projection.lora_A"][0, 0] = float("nan")
    elif change == "dtype": value["smolvlm2_lora"]["projection.lora_A"] = value["smolvlm2_lora"]["projection.lora_A"].bfloat16()
    elif change == "unpaired": value["smolvlm2_lora"].pop("projection.lora_B")
    else: value["lora_spec"] = {**value["lora_spec"], "rank": 8}
    with pytest.raises(ValueError): runner.check_checkpoint(value, "job_b")


def test_actual_job_b_loader_installs_all_adapters_and_uses_them_for_text(tmp_path, monkeypatch):
    from tools import infer_mobileov_stage1 as infer

    config, smol = tmp_path / "config", tmp_path / "smol"
    config.write_text("fixture"); smol.write_text("fixture weights")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=smol))
    def stack(args, device):
        torch.manual_seed(7)
        frozen = nn.Module()
        frozen.backbone = Backbone().bfloat16().requires_grad_(False).eval()
        frozen.processor, frozen.processor_sha256 = Processor(), "proc"
        frozen.max_tokens, frozen.input_dim = 2048, 32
        return cfg, frozen, Stage1Connector(32, 16, 12), nn.Linear(12, 1).bfloat16(), nn.Linear(1, 1).bfloat16(), None
    _, frozen, connector, dit, vae, _ = stack(None, None)
    signatures = frozen_signatures(dict(smolvlm2=frozen, dit=dit, vae=vae))
    owner = job.LoRAEncoder(frozen, gradient_checkpointing=False)
    with torch.no_grad():
        for name, parameter in owner.named_parameters():
            if name.endswith("lora_B"): parameter.fill_(.01)
    value = payload("job_b")
    training = value["contract"]
    parent = training["parent_contract"]
    for target in (training, parent):
        target.update(config_sha256=protocol.sha256(config), processor="fixture", processor_sha256="proc", max_tokens=2048,
                      frozen_weights_sha256={key: signatures[key] for key in ("smolvlm2", "vae")})
    parent.update(initial_weights_sha256=signatures, smolvlm2_sha256=protocol.sha256(smol))
    value.update(connector_spec=connector.spec, connector=deepcopy(connector.state_dict()), dit=deepcopy(dit.float().state_dict()),
                 smolvlm2_lora=job.lora_state(owner))
    monkeypatch.setattr(infer, "load_stack", stack)
    args = SimpleNamespace(config=config)
    loaded, loaded_connector, loaded_dit, loaded_vae, _, audit = runner.load_job_b_models(args, value, torch.device("cpu"))
    assert audit["loaded_lora_exact"] and audit["trained_lora_loaded"]
    assert all(torch.equal(tensor, value["smolvlm2_lora"][key]) for key, tensor in job.lora_state(loaded).items())
    assert next(loaded_dit.parameters()).dtype == torch.bfloat16
    assert all(not p.requires_grad for module in (loaded, loaded_connector, loaded_dit, loaded_vae) for p in module.parameters())
    _, baseline, _, _, _, _ = stack(None, None)
    baseline = job.LoRAEncoder(baseline, gradient_checkpointing=False).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        before, _ = baseline("A small red car.")
        after, _ = loaded("A small red car.")
        negative, _ = loaded("", drop_condition=True)
    assert any(not torch.equal(a, b) for a, b in zip(before, after)) and negative
    missing = deepcopy(value)
    for key in list(missing["smolvlm2_lora"]):
        if ".q_proj." in key: del missing["smolvlm2_lora"][key]
    with pytest.raises(ValueError, match="LoRA checkpoint keys"):
        runner.load_job_b_models(args, missing, torch.device("cpu"))


def test_cli_selects_only_unique_completed_history_and_separates_smokes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HISTORY_ROOT", tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        runner.parse_args(["--experiment", "uniform_history"])
    run = tmp_path / "123"
    protocol.atomic_json(run / "training_complete.json", {"step": 378636})
    torch.save(payload("uniform_history"), run / "stage2_dit_connector_latest.pt")
    args = runner.parse_args(["--experiment", "uniform_history"])
    assert args.checkpoint.parent == run and args.output_dir.name == "vbench_mobileov_uniform_history_5x_dual"
    for extra in (["--allow-incomplete-checkpoint"], ["--samples-per-prompt", "1"]):
        with pytest.raises(SystemExit): runner.parse_args(["--experiment", "job_b", *extra])
    args = runner.parse_args(["--experiment", "job_b", "--max-prompts", "2", "--samples-per-prompt", "1"])
    assert args.checkpoint == runner.DEFAULT_JOB_B and args.output_dir.name.endswith("_1x_dual_smoke2")
    protocol.atomic_json(tmp_path / "456/training_complete.json", {"step": 378636})
    (tmp_path / "456/stage2_dit_connector_latest.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="exactly one"): runner.find_history_checkpoint()


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
def test_preflight_preserves_parent_protocol_and_pins_new_code(tmp_path, monkeypatch, experiment):
    root, parent, _, _ = reference_fixture(tmp_path)
    value = deepcopy(parent)
    child = payload(experiment)
    value.update(system=runner.SYSTEMS[experiment], training_contract=child["contract"],
                 checkpoint=dict(path="new.pt", sha256="a" * 64, step=378636, final_step=378636))
    def prepare(args):
        assert runner.base.SYSTEM == runner.SYSTEMS[experiment]
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "sample_plan.json", dict(samples=list(runner.base.items_for(value))))
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", {})
        return deepcopy(value)
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(experiment=experiment, output_dir=tmp_path / "new", reference_run=root, max_prompts=0)
    contract = runner.prepare(args)
    assert runner.prepare(args) == contract
    assert contract["continuation_evaluation"]["reference_verified"]
    assert contract["continuation_evaluation"]["lora_required"] == (experiment == "job_b")
    assert not contract["continuation_evaluation"]["history_corruption_at_inference"]
    assert len(protocol.read_json(args.output_dir / "sample_plan.json")["samples"]) == 4720
    assert all(name in contract["implementation_sha256"] for name in runner.IMPLEMENTATION)
    value["seed"] += 1
    with pytest.raises(ValueError, match="Evaluation differs"):
        runner.prepare(args)


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
def test_worker_exports_real_paired_mp4_and_reuses_cache(tmp_path, monkeypatch, experiment):
    from tools import infer_mobileov_stage1 as infer
    from new_mobile_ov.generation import quicksrnet

    frames, kwargs = fixture_pair(tmp_path)
    contract = kwargs["contract"]
    path = tmp_path / "final.pt"
    torch.save(payload(experiment), path)
    contract.update(system=runner.SYSTEMS[experiment], training_complete=True,
                    checkpoint=dict(path=str(path), sha256=protocol.sha256(path), step=378636),
                    continuation_evaluation=dict(experiment=experiment, final_step=378636))
    protocol.atomic_json(tmp_path / "run_contract.json", contract)
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-integration-only")
    calls = []
    def encoder(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return torch.full((1, 2, 3), 2.), torch.ones(1, 2), torch.ones(1, 4)
    def load_models(args, value, device):
        assert value["step"] == 378636
        return encoder, lambda *parts: parts, None, None, None, dict(trained_lora_loaded=experiment == "job_b")
    if experiment == "job_b": monkeypatch.setattr(runner, "load_job_b_models", load_models)
    else: monkeypatch.setattr(runner.base.stage2, "load_models", load_models)
    generated = []
    def generate(*args, **kw):
        generated.append(kw)
        assert "anchor_latent" not in kw and "history_corrupt_max" not in kw
        assert (kw["first_steps"], kw["video_steps"], kw["guidance"], kw["video_guidance"]) == (20, 10, 7, 5)
        assert bool((args[3][0] == 2).all())
        return frames, [{"unit": 0}]
    monkeypatch.setattr(infer, "autocast", lambda _: nullcontext())
    monkeypatch.setattr(infer, "generate_frames", generate)
    monkeypatch.setattr(quicksrnet, "load_quicksrnet", lambda **kw: (torch.nn.Upsample(scale_factor=2), None))
    args = SimpleNamespace(output_dir=tmp_path, rank=0, workers=1)
    runner.generation_worker(args, device=torch.device("cpu"))
    runner.generation_worker(args, device=torch.device("cpu"))
    assert len(generated) == 1 and calls == [("", {"drop_condition": True}), ("panda", {})]
    with runner.continuation_policy(experiment): runner.base.verify_generation(tmp_path)
    raw = protocol.read_json(tmp_path / "no_quicksr/video_metadata/0000_0.json")
    sr = protocol.read_json(tmp_path / "quicksr/video_metadata/0000_0.json")
    assert raw["base_frames_sha256"] == sr["base_frames_sha256"]
    assert raw["first_frame_generated"] and not raw["external_anchor"]


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
def test_summary_and_parent_comparison_include_both_variants_and_all_dimensions(tmp_path, monkeypatch, experiment):
    root, parent, report, assets = reference_fixture(tmp_path)
    value = deepcopy(parent)
    value.update(system=runner.SYSTEMS[experiment], checkpoint=dict(step=378636), scope="test matched continuation",
                 continuation_evaluation=dict(experiment=experiment, final_step=378636, reference_verified=True,
                     reference_run=str(root), reference_contract_sha256=protocol.fingerprint(parent),
                     reference_report_sha256=protocol.fingerprint(report)))
    output = tmp_path / "new"
    protocol.atomic_json(output / "run_contract.json", value)
    protocol.atomic_json(output / "score_assets.json", assets)
    score = dict(quality_score=.8, semantic_score=.6, total_score=.76, raw={d:.5 for d in runner.base.DIMENSIONS})
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = lambda _: deepcopy(score)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.base, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.base.subprocess, "run", lambda *a, **kw: None)
    for variant in runner.base.VARIANTS:
        for dimension in runner.base.DIMENSIONS:
            protocol.atomic_json(output / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json",
                                 {dimension: [.5, []]})
    args = SimpleNamespace(output_dir=output)
    runner.summarize(args)
    result = protocol.read_json(output / f"{experiment}_vs_epoch3.json")
    assert len(result["rows"]) == 19
    assert set(result["rows"]["total_score"]) == {"no_quicksr", "quicksr"}
    assert "no external anchor" in (output / "quicksr_ablation.md").read_text()
    value["max_prompts"] = 1
    protocol.atomic_json(output / "run_contract.json", value)
    with pytest.raises(ValueError, match="No full score"): runner.summarize(args)


@pytest.mark.parametrize("experiment", runner.SYSTEMS)
def test_berzelius_launchers_choose_correct_checkpoint_and_keep_outputs_separate(launch, monkeypatch, experiment):
    script = ROOT / f"scripts/vbench_mobileov_{'job_b' if experiment == 'job_b' else 'uniform_history'}_1node8gpu.sbatch"
    monkeypatch.setattr(launch_helpers, "SCRIPT", script)
    root, env, run = launch
    (root / "scripts/run_vbench_mobileov_continuation.sh").write_text((ROOT / "scripts/run_vbench_mobileov_continuation.sh").read_text())
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
out=pathlib.Path(argv[argv.index("--output-dir")+1]);out.mkdir(parents=True,exist_ok=True)
experiment=argv[argv.index("--experiment")+1]
(out/(experiment+"_vs_epoch3.json")).write_text("{}")
(out/(experiment+"_vs_epoch3.md")).write_text("mock comparison")
''')
    reference = root / "output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(reference / name, {})
    train = root / ("output/mobileov_stage2_job_b_lora/17783291" if experiment == "job_b"
                    else "output/mobileov_stage2_uniform_history/12345")
    protocol.atomic_json(train / "training_complete.json", {"step": 378636})
    filename = "job_b_dit_connector_lora_latest.pt" if experiment == "job_b" else "stage2_dit_connector_latest.pt"
    checkpoint = train / filename
    checkpoint.write_bytes(b"fixture final weights")
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert option(argv, "--experiment") == experiment and option(argv, "--workers") == "8"
    assert root / option(argv, "--checkpoint") == checkpoint
    assert option(argv, "--samples-per-prompt") == "5" and "--hf-file" not in argv
    assert "mock comparison" in result.stdout
    result, argv = run(MAX_PROMPTS="2", SAMPLES_PER_PROMPT="1")
    assert result.returncode == 0 and option(argv, "--output-dir").endswith("_1x_dual_smoke2")
    assert run(SAMPLES_PER_PROMPT="1")[0].returncode != 0
    assert run(CHECKPOINT="missing.pt")[0].returncode != 0
    if experiment == "uniform_history":
        protocol.atomic_json(train.parent / "other/training_complete.json", {"step": 378636})
        assert run()[0].returncode != 0
        assert run(TRAIN_RUN=str(train))[0].returncode == 0
        assert run(CHECKPOINT=str(checkpoint))[0].returncode == 0
    for path in (script, ROOT / "scripts/run_vbench_mobileov_continuation.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_help_works_outside_the_repo(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/benchmark_mobileov_continuations_dual.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--experiment" in result.stdout and "--reference-run" in result.stdout


def test_gpu_history_smoke_resume_stays_inside_tiny_dataset_budget():
    from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan
    script = (ROOT / "scripts/smoke_mobileov_stage2_history_local.sbatch").read_text()
    pause, finish = map(int, re.findall(r"--stop-after (\d+)", script))
    plan = MediaEpochPlan(3, 5, 1, 2, epochs=1, start_epoch=3)
    assert runner.PARENT_STEP < pause < finish == runner.PARENT_STEP + plan.steps
