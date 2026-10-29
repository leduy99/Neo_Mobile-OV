from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from tools import benchmark_mobileov_stage2_ssd1b_checkpoint_sweep as runner
import test_stage2_resume_epoch4_vbench as clean
import test_stage2_history_fresh_epochs_vbench as history
import test_stage2_history_fresh_epoch5_vbench as history5
import test_stage2_ssd1b_conditioning_vbench as old_anchor
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

protocol, ROOT = clean.protocol, clean.ROOT
SCRIPT = ROOT / "scripts/vbench_mobileov_stage2_ssd1b_checkpoint_sweep_1node8gpu.sbatch"


def payload(case):
    spec = runner.CASES[case]
    if spec["kind"] == "resume":
        return dict(clean.payload(), step=spec["step"])
    return history5.payload() if spec["epoch"] == 5 else history.payload(spec["epoch"])


def write_reference(folder, contract, assets, score):
    report = dict(status="complete", protocol=contract, evaluation_assets=assets,
        scores={variant: {contract["system"]: deepcopy(score)} for variant in runner.base.VARIANTS})
    for name, value in (("run_contract.json", contract), ("quicksr_ablation.json", report), ("score_assets.json", assets)):
        protocol.atomic_json(folder / name, value)
    return report


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    _, original, old_report, assets = clean.reference_fixture(tmp_path / "old")
    original["implementation_sha256"] = {name: protocol.sha256(ROOT / name) for name in runner.NUMERICAL_FILES}
    original["exports"]["quicksr"].update(width=1024, height=640)
    original["conditioning"] = "literal prompt; trained empty negative; no modifier"
    score = old_report["scores"]["quicksr"][original["system"]]
    images = dict(format=runner.shared.FORMAT, prompts=original["prompts"], max_prompts=0,
        samples_per_prompt=5, full_info_sha256=original["full_info_sha256"], seed=2026, seed_mapping="extendable",
        config_sha256=original["config_sha256"], library_versions=original["library_versions"],
        vbench_source_commit=original["vbench_source_commit"], admission_checkpoint_sha256=runner.shared.FINAL_SHA256,
        image=dict(width=1024, height=640, image_bridge=False))
    manifest = [dict(key="0000_0", sha256="image")]
    frames = tmp_path / "shared"
    protocol.atomic_json(frames / "run_contract.json", images)
    protocol.atomic_json(frames / "image_manifest.json", manifest)
    baseline = deepcopy(original)
    baseline.update(format=runner.anchor.FORMAT, mode=runner.MODE, system="stage2_ssd1b_anchor",
        external_anchor=True, image_to_smolvlm2=False, first_frame_generated=False,
        shared_ssd1b=dict(path=str(frames), contract_sha256=protocol.fingerprint(images),
                          image_manifest_sha256=protocol.fingerprint(manifest)),
        image_processing=dict(width=512, height=320, resampling="PIL.LANCZOS", pillow_version=runner.version("Pillow")),
        vae_seed_policy="video seed + 228; private generator; video noise reset independently",
        conditioning="literal text ONLY through SmolVLM2; empty negative; shared SSD1B image through VAE ONLY")
    baseline["implementation_sha256"].update({name: protocol.sha256(ROOT / name) for name in runner.ANCHOR_FILES})
    anchor_root = tmp_path / "anchor_reference"
    write_reference(anchor_root, baseline, assets, score)
    cases, templates = deepcopy(runner.CASES), {}
    for case, spec in cases.items():
        weight = tmp_path / "weights" / f"{case}.pt"
        weight.parent.mkdir(exist_ok=True)
        value = payload(case)
        torch.save(value, weight)
        reference = deepcopy(original)
        reference.update(system=spec["system"], training_contract=value["contract"],
            training_complete=spec["step"] == 473295,
            checkpoint=dict(path=str(weight), sha256=protocol.sha256(weight), step=spec["step"], final_step=473295))
        text_root = tmp_path / "text_controls" / case
        write_reference(text_root, reference, assets, score)
        templates[str(weight)] = reference
        spec.update(checkpoint=str(weight), text_reference=str(text_root))
    monkeypatch.setattr(runner, "CASES", cases)
    monkeypatch.setattr(runner.shared, "verify_cache", lambda path: (deepcopy(images), deepcopy(manifest)))
    def prepare(args):
        template = deepcopy(templates[str(args.checkpoint)])
        with runner.case_policy(next(case for case, spec in cases.items() if spec["checkpoint"] == str(args.checkpoint))):
            value, _, target = runner.base.read_checkpoint(args.checkpoint)
        assert value["step"] == template["checkpoint"]["step"] and target == 473295
        template["system"] = runner.base.SYSTEM
        protocol.atomic_json(args.output_dir / "vbench_info.json", [])
        protocol.atomic_json(args.output_dir / "checkpoint_audit.json", dict(training_complete=template["training_complete"]))
        return template
    monkeypatch.setattr(runner.base, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "sweep", frames_root=None, anchor_reference=anchor_root,
        cases=list(cases), max_prompts=0, samples_per_prompt=5, seed=2026, allow_incomplete_checkpoint=False,
        expected_sha256=None)
    suite = runner.prepare_suite(args)
    return args, suite, baseline, assets, score


def test_cli_defaults_one_sweep_with_five_distinct_checkpoints():
    args = runner.parse_args([])
    assert args.cases == list(runner.CASES) and args.workers == 8 and args.samples_per_prompt == 5
    assert args.output_dir == runner.DEFAULT_OUTPUT and args.frames_root is None
    assert args.case is None and args.mode == "ssd1b_anchor"
    for flags in (["--cases", "resume_epoch4", "resume_epoch4"], ["--case", "resume_epoch4"],
                  ["--phase", "generation-worker"], ["--checkpoint", "/old.pt"], ["--allow-incomplete-checkpoint"]):
        with pytest.raises(SystemExit):
            runner.parse_args(flags)
    worker = runner.parse_args(["--phase", "generation-worker", "--case", "history_epoch5", "--output-dir", "/tmp/sweep"])
    assert worker.output_dir == Path("/tmp/sweep/history_epoch5") and worker.suite_root == Path("/tmp/sweep")
    subset = runner.parse_args(["--cases", "history_epoch3", "history_epoch5", "--max-prompts", "2", "--samples-per-prompt", "1"])
    assert subset.output_dir.name.endswith("_1x_dual_smoke2")


@pytest.mark.parametrize("case", runner.CASES)
def test_admission_keeps_exact_step_history_recipe_and_legacy_policy(case):
    value = payload(case)
    original = deepcopy(value["contract"])
    before = runner.base.SYSTEM, runner.base.check_media_checkpoint, deepcopy(runner.anchor.MODES)
    with runner.case_policy(case):
        assert runner.check_checkpoint(value, case) == 473295
        assert runner.base.SYSTEM == runner.selected_system(case)
        assert runner.anchor.route(runner.MODE) == dict(system=runner.selected_system(case), image_to_smolvlm2=False, external_anchor=True)
        with pytest.raises(ValueError):
            runner.check_checkpoint(dict(value, step=1), case, allow_incomplete=True)
        with pytest.raises(ValueError):
            runner.check_checkpoint(dict(value, optimizer={}), case)
    assert before == (runner.base.SYSTEM, runner.base.check_media_checkpoint, runner.anchor.MODES)
    assert value["contract"] == original


def test_prepare_reuses_cache_and_baseline_without_relabeling_or_mutating_them(prepared):
    args, suite, baseline, _, _ = prepared
    before = {str(path): protocol.sha256(path) for path in args.frames_root.rglob("*.json")}
    baseline_sha = protocol.sha256(args.anchor_reference / "quicksr_ablation.json")
    assert runner.prepare_suite(args) == suite
    assert suite["image_generation"] is False and len(suite["cases"]) == 5
    for case, digest in suite["cases"].items():
        contract = runner.checked_case(args.output_dir / case, case)
        assert protocol.fingerprint(contract) == digest
        assert contract["external_anchor"] is True and contract["image_to_smolvlm2"] is False
        assert contract["shared_ssd1b"]["contract_sha256"] == baseline["shared_ssd1b"]["contract_sha256"]
        assert contract["checkpoint"]["sha256"] != runner.shared.FINAL_SHA256
        assert contract["training_complete"] is (runner.CASES[case]["step"] == 473295)
        assert "parent_contract" not in contract["training_contract"]
        assert len(protocol.read_json(args.output_dir / case / "sample_plan.json")["samples"]) == 4720
    assert before == {str(path): protocol.sha256(path) for path in args.frames_root.rglob("*.json")}
    assert baseline_sha == protocol.sha256(args.anchor_reference / "quicksr_ablation.json")


@pytest.mark.parametrize("change", ["seed", "config_sha256", "library_versions", "vbench_source_commit", "admission", "prompts"])
def test_incompatible_shared_images_are_rejected(prepared, change):
    args, _, _, _, _ = prepared
    images = protocol.read_json(args.frames_root / "run_contract.json")
    template = deepcopy(protocol.read_json(args.output_dir / "resume_epoch4/run_contract.json"))
    runner.validate_image_protocol(template, images)
    if change == "admission": images["admission_checkpoint_sha256"] = "wrong"
    elif change == "prompts": images["prompts"][0] = "wrong"
    else: images[change] = "wrong"
    with pytest.raises(ValueError):
        runner.validate_image_protocol(template, images)


def test_smoke_can_reuse_prefix_of_full_image_cache(prepared):
    args, _, _, _, _ = prepared
    images = protocol.read_json(args.frames_root / "run_contract.json")
    template = protocol.read_json(args.output_dir / "resume_epoch4/run_contract.json")
    template.update(prompts=template["prompts"][:2], max_prompts=2, samples_per_prompt=1)
    runner.validate_image_protocol(template, images)
    template["prompts"].reverse()
    with pytest.raises(ValueError, match="exact prompts"):
        runner.validate_image_protocol(template, images)


def test_control_weight_changes_and_output_collisions_fail_before_generation(prepared):
    args, _, _, _, _ = prepared
    root = Path(runner.CASES["resume_epoch4"]["text_reference"])
    contract, report, assets = runner.read_reference(root)
    contract["checkpoint"]["sha256"] = "wrong"
    write_reference(root, contract, assets, report["scores"]["quicksr"][contract["system"]])
    with pytest.raises(ValueError, match="exact weights"):
        runner.prepare_suite(args)
    for output in (args.frames_root, args.frames_root / "child", args.frames_root.parent, args.anchor_reference):
        with pytest.raises(ValueError, match="separate"):
            runner.prepare_suite(SimpleNamespace(**dict(vars(args), output_dir=output)))


def test_changed_numerical_code_or_reference_breaks_comparison(prepared):
    args, _, _, assets, _ = prepared
    contract = runner.checked_case(args.output_dir / "resume_epoch4", "resume_epoch4")
    text, baseline = runner.verified_controls(contract)
    current = deepcopy(contract)
    current["implementation_sha256"][runner.ANCHOR_FILES[0]] = "changed"
    with pytest.raises(ValueError, match="Numerical"):
        runner.compare_numerical_protocol(baseline[0], current, anchored=True)
    current = deepcopy(contract)
    current["image_processing"]["pillow_version"] = "changed"
    with pytest.raises(ValueError, match="preprocessing"):
        runner.compare_numerical_protocol(baseline[0], current, anchored=True)
    protocol.atomic_json(args.output_dir / "resume_epoch4/score_assets.json", [dict(assets[0], sha256="wrong")])
    with pytest.raises(ValueError, match="evaluator weights"):
        runner.verify_score_assets(args.output_dir / "resume_epoch4", contract)
    report = deepcopy(text[1]); report["scores"]["quicksr"][text[0]["system"]]["total_score"] = .1
    protocol.atomic_json(Path(contract["ssd1b_checkpoint_sweep"]["text_reference"]) / "quicksr_ablation.json", report)
    with pytest.raises(ValueError, match="reference changed"):
        runner.verified_controls(contract)


def test_summaries_cover_intermediate_archives_both_variants_and_reused_baseline(prepared, monkeypatch):
    args, suite, _, assets, score = prepared
    module = ModuleType("evaluate_vbench_resume")
    def tabulate(folder):
        value = deepcopy(score)
        value["total_score"] += .04 if "quicksr" in folder.parts else .03
        return value
    module.tabulate = tabulate
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner.anchor, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    for case in suite["cases"]:
        output = args.output_dir / case
        protocol.atomic_json(output / "score_assets.json", assets)
        for variant in runner.base.VARIANTS:
            for dimension in runner.base.DIMENSIONS:
                protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                     f"{dimension}_eval_results.json", {dimension: [.5, []]})
        runner.summarize_case(SimpleNamespace(case=case, output_dir=output))
    runner.summarize_suite(args)
    report = protocol.read_json(args.output_dir / "comparison.json")
    assert report["status"] == "complete" and len(report["rows"]) == 12
    assert sum("reused" in row["setting"] for row in report["rows"]) == 2
    for row in report["rows"][2:]:
        assert row["anchor_effect_pp"] == pytest.approx(4. if row["variant"] == "quicksr" else 3.)
    assert "Anchor vs Same-Checkpoint Text-Only" in (args.output_dir / "comparison.md").read_text()
    assert "spatial_relationship" in (args.output_dir / "dimensions.md").read_text()
    assert protocol.read_json(args.output_dir / "resume_epoch4/quicksr_ablation.json")["protocol"]["training_complete"] is False


def test_one_allocation_runs_five_cases_sequentially_on_eight_workers(tmp_path, monkeypatch):
    args = SimpleNamespace(phase="run", case=None, output_dir=tmp_path / "sweep", workers=8,
        max_prompts=0, vbench_python=sys.executable, vbench_repo=tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare_suite", lambda _: dict(cases=dict.fromkeys(runner.CASES)))
    monkeypatch.setattr(runner, "checked_case", lambda *a: {})
    monkeypatch.setattr(runner, "verify_score_assets", lambda *a: None)
    monkeypatch.setattr(runner.anchor, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.base.core, "gpu_tokens", lambda _: list(map(str, range(8))))
    calls = []
    def run(argv, **kw):
        calls.append((option(argv, "--phase"), option(argv, "--case") if "--case" in argv else None))
        if option(argv, "--phase") == "warm-scores":
            protocol.atomic_json(args.output_dir / "score_assets.json", [])
    def workers(commands, devices, folder, **kw):
        assert len(commands) == len(devices) == 8
        assert {option(argv, "--rank") for argv in commands} == set(map(str, range(8)))
        case = option(commands[0], "--case")
        assert all(option(argv, "--case") == case for argv in commands)
        calls.append((option(commands[0], "--phase"), case))
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.base.core, "run_processes", workers)
    runner.main()
    assert calls == [("warm-scores", None)] + [item for case in runner.CASES for item in
        (("generation-worker", case), ("score-worker", case), ("summarize", case))] + [("summarize", None)]


@pytest.mark.parametrize("case", runner.CASES)
def test_actual_anchor_worker_reuses_old_pixels_text_only_and_vae_route(tmp_path, monkeypatch, case):
    from tools import infer_mobileov_stage1 as infer
    from new_mobile_ov.generation import quicksrnet

    frames_root = tmp_path / "shared"
    _, image = old_anchor.make_shared(frames_root)
    with runner.case_policy(case):
        contract = old_anchor.video_contract(frames_root, runner.MODE)
    spec = runner.CASES[case]
    contract.update(system=runner.selected_system(case), training_complete=spec["step"] == 473295,
        first_frame_generated=False, checkpoint=dict(path="unused", sha256="trained", step=spec["step"], final_step=473295),
        ssd1b_checkpoint_sweep=dict(case=case, eligible_archive=True, image_generation=False, history_noise_at_inference=False))
    output = tmp_path / "sweep" / case
    quicksr = tmp_path / "sr.pt"; quicksr.write_bytes(b"test sr")
    contract["quicksr"] = dict(path=str(quicksr), sha256=protocol.sha256(quicksr))
    protocol.atomic_json(output / "run_contract.json", contract)
    protocol.atomic_json(output.parent / "run_contract.json", dict(format=runner.FORMAT, cases={case: protocol.fingerprint(contract)}))
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    calls = []
    def encoder(*args, **kw):
        assert len(args) == 1 and "allow_text_image" not in kw
        calls.append("text")
        return torch.zeros(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 4)
    monkeypatch.setattr(runner.base, "read_checkpoint", lambda *a, **kw: ({"step": spec["step"]}, "trained", 473295))
    monkeypatch.setattr(runner.base.stage2, "load_models", lambda *a: (encoder, lambda *v: v, None, None, None, {}))
    monkeypatch.setattr(infer, "autocast", lambda _: nullcontext())
    def encode(vae, video, generator, **kw):
        assert tuple(video.shape) == (1, 3, 1, 320, 512) and generator.initial_seed() == 2026 + 228
        calls.append("VAE")
        return "latent"
    monkeypatch.setattr(infer, "encode_anchor", encode)
    def generate(*a, **kw):
        assert kw["anchor_latent"] == "latent" and kw["seed"] == 2026
        assert (kw["first_steps"], kw["video_steps"], kw["guidance"], kw["video_guidance"]) == (20, 10, 7., 5.)
        calls.append("generate")
        return [Image.new("RGB", (32, 16), "red")] * 3, [dict(unit=0, supplied_anchor=True)]
    monkeypatch.setattr(infer, "generate_frames", generate)
    monkeypatch.setattr(quicksrnet, "load_quicksrnet", lambda **kw: (torch.nn.Upsample(scale_factor=2), None))
    monkeypatch.setattr(runner.shared, "load_native_ssd", lambda *a: pytest.fail("Must never load/generate SSD1B"))
    args = SimpleNamespace(case=case, output_dir=output, rank=0, workers=1, mode=runner.MODE)
    image_sha = protocol.sha256(image)
    runner.generation_worker(args, device=torch.device("cpu"))
    assert calls == ["text", "text", "VAE", "generate"]
    assert protocol.sha256(image) == image_sha
    before = list(calls)
    runner.generation_worker(args, device=torch.device("cpu"))
    assert calls == before
    records = [protocol.read_json(output / variant / "video_metadata/0000_0.json") for variant in runner.base.VARIANTS]
    assert all(record["shared_image_sha256"] == image_sha for record in records)
    assert records[0]["base_frames_sha256"] == records[1]["base_frames_sha256"]


def test_slurm_submits_one_allocation_and_never_calls_image_preparation(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, env, run = launch
    reference = root / "output/vbench_stage2_epoch3_ssd1b_conditioning_5x_dual/ssd1b_anchor"
    protocol.atomic_json(reference / "quicksr_ablation.json", {})
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
(output/"comparison.json").write_text("{}")
(output/"comparison.md").write_text("mock sweep report")
''')
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert "tools/benchmark_mobileov_stage2_ssd1b_checkpoint_sweep.py" in argv
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--output-dir").endswith("checkpoint_ssd1b_sweep_5x_dual")
    assert "--frames-root" not in argv and "--cases" not in argv
    assert "mock sweep report" in result.stdout
    result, argv = run(SWEEP_CASES="resume_epoch5 history_epoch5", ANCHOR_ROOT="/custom/cache")
    assert result.returncode == 0, result.stderr
    assert argv[argv.index("--cases") + 1:argv.index("--frames-root")] == ["resume_epoch5", "history_epoch5"]
    assert option(argv, "--frames-root") == "/custom/cache"
    source = SCRIPT.read_text()
    assert source.count("srun --") == 1 and "sbatch " not in source
    assert "#SBATCH --job-name=mov-sanc" in source and "#SBATCH --gres=gpu:8" in source
    assert "prepare_vbench_stage2_ssd1b_frames.py" not in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("phase", ["warm-scores", "score-worker", "summarize-case", "summarize-suite", "help"])
def test_sweep_scoring_processes_do_not_import_av_or_training_stack(prepared, phase):
    args, suite, _, assets, score = prepared
    protocol.atomic_json(args.output_dir / "fixture_score.json", score)
    for case in suite["cases"]:
        output = args.output_dir / case
        contract = runner.checked_case(output, case)
        write_reference(output, contract, assets, score)
        for variant in runner.base.VARIANTS:
            for dimension in runner.base.DIMENSIONS:
                protocol.atomic_json(output / variant / "scores" / dimension / "vbench" /
                                     f"{dimension}_eval_results.json", {dimension: [.5, []]})
    program = r'''
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType
script, phase, root = sys.argv[1:]
root = Path(root)
class NoAV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise ModuleNotFoundError("No module named 'av' (isolated VBench simulation)")
sys.meta_path.insert(0, NoAV())
scope = runpy.run_path(script, run_name="isolated_vbench")["main"].__globals__
events = []
scope["base"].core.warm_scores = lambda args: events.append("warm-scores")
scope["anchor"].verify_generation = lambda output: None
scope["subprocess"].run = lambda argv, **kw: events.append(argv[1])
module = ModuleType("evaluate_vbench_resume")
module.tabulate = lambda folder: json.loads((root / "fixture_score.json").read_text())
sys.modules["evaluate_vbench_resume"] = module
flags = ["--output-dir", str(root)]
if phase == "help":
    flags += ["--help"]
elif phase == "summarize-case":
    flags += ["--phase", "summarize", "--case", "resume_epoch4"]
elif phase == "summarize-suite":
    flags += ["--phase", "summarize"]
else:
    flags += ["--phase", phase]
    if phase == "score-worker": flags += ["--case", "resume_epoch4"]
sys.argv = [script, *flags]
try:
    scope["main"]()
except SystemExit as error:
    assert phase == "help" and error.code == 0
if phase == "warm-scores":
    assert events == ["warm-scores"]
elif phase == "score-worker":
    assert events and all(name.endswith("evaluate_vbench_resume.py") for name in events)
elif phase == "summarize-case":
    assert (root / "resume_epoch4/quicksr_ablation.json").is_file()
elif phase == "summarize-suite":
    assert (root / "comparison.md").is_file()
assert not any(name.startswith("new_mobile_ov.training") for name in sys.modules)
print("ISOLATED_SWEEP_OK")
'''
    result = subprocess.run([sys.executable, "-c", program,
        str(ROOT / "tools/benchmark_mobileov_stage2_ssd1b_checkpoint_sweep.py"), phase, str(args.output_dir)],
        cwd=args.output_dir, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATED_SWEEP_OK" in result.stdout
