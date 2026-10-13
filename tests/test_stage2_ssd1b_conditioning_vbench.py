from contextlib import nullcontext
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tools import benchmark_stage2_ssd1b_conditioning_dual as runner

shared, base = runner.shared, runner.base
protocol = sys.modules["vbench_joint_native_protocol"]
ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "frames": "prepare_vbench_stage2_ssd1b_frames_1node8gpu.sbatch",
    "ssd1b_anchor": "vbench_stage2_epoch3_ssd1b_anchor_1node8gpu.sbatch",
    "image_condition": "vbench_stage2_epoch3_image_condition_1node8gpu.sbatch",
}


def shared_contract():
    return dict(format=shared.FORMAT, prompts=["panda"], samples_per_prompt=1, seed=2026,
                max_prompts=1, full_info_sha256="full", config_sha256="cfg", seed_mapping="extendable",
                admission_checkpoint_sha256=shared.FINAL_SHA256, vbench_source_commit="git",
                image=dict(width=1024, height=640, prompt_modifier=" native suffix"),
                library_versions={name: version(name) for name in ("torch", "diffusers", "transformers")},
                implementation_sha256={}, model_path="unused", native_repo="unused",
                native_files=[], native_source_sha256={})


def make_shared(path):
    value = shared_contract()
    protocol.atomic_json(path / "run_contract.json", value)
    item = next(shared.items_for(value))
    image = path / "images" / f"{item['key']}.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1024, 640), "red").save(image)
    protocol.atomic_json(path / "records" / f"{item['key']}.json", dict(
        sample=item, sha256=protocol.sha256(image), contract_sha256=protocol.fingerprint(value)))
    shared.summarize(path)
    return value, image


def video_contract(path, mode):
    images, manifest = shared.verify_cache(path)
    return dict(format=runner.FORMAT, mode=mode, **runner.route(mode), prompts=images["prompts"],
                samples_per_prompt=1, seed=2026, max_prompts=1, training_complete=True,
                checkpoint=dict(path="unused", sha256=shared.FINAL_SHA256, step=283977),
                config="unused", training_contract={}, implementation_sha256={},
                image_processing=dict(pillow_version=version("Pillow")), library_versions=images["library_versions"],
                shared_ssd1b=dict(path=str(path), contract_sha256=protocol.fingerprint(images),
                                 image_manifest_sha256=protocol.fingerprint(manifest)),
                exports={"no_quicksr":dict(width=32, height=16, frames=3, fps=24),
                         "quicksr":dict(width=64, height=32, frames=3, fps=24)})


def make_pair(path, mode="ssd1b_anchor"):
    _, image = make_shared(path / "shared")
    value = video_contract(path / "shared", mode)
    output = path / mode
    protocol.atomic_json(output / "run_contract.json", value)
    item = next(base.items_for(value))
    kwargs = dict(output=output, contract=value, item=item, model=torch.nn.Upsample(scale_factor=2),
                  device=torch.device("cpu"), statistics=[dict(unit=0, supplied_anchor=value["external_anchor"])],
                  image_sha=protocol.sha256(image), pixels_sha="a"*64, condition_sha="b"*64)
    frames = [Image.new("RGB", (32, 16), "red")] * 3
    runner.export_pair(frames, **kwargs)
    return frames, kwargs


def test_shared_plan_matches_video_seeds_and_all_eight_shards():
    value = dict(prompts=[f"prompt {i}" for i in range(944)], samples_per_prompt=5, seed=2026)
    images = list(shared.items_for(value))
    videos = list(base.items_for(value))
    assert len(images) == len({i["anchor_seed"] for i in images}) == 4720
    assert images == [runner.frame_item(i) for i in videos]
    shards = [protocol.shard_items(images, rank, 8) for rank in range(8)]
    assert len({i["key"] for shard in shards for i in shard}) == 4720
    assert images[0]["anchor_seed"] == 2026 and images[5]["anchor_seed"] == 2027


@pytest.mark.parametrize("mode", runner.MODES)
def test_condition_routing_never_combines_the_two_image_paths(mode):
    calls = []
    image = Image.new("RGB", (512, 320))
    def encoder(*args, **kwargs):
        calls.append((args, kwargs))
        return ("tokens", "mask", "pooled")
    result = runner.positive_condition(encoder, lambda *v: v, "literal text", image, mode)
    assert result == ("tokens", "mask", "pooled")
    if mode == "ssd1b_anchor":
        assert calls == [(("literal text",), {})]
    else:
        assert calls == [(("literal text", image), {"allow_text_image":True})]
        with pytest.raises(ValueError, match="requires"):
            runner.positive_condition(encoder, lambda *v: v, "p", None, mode)
    with pytest.raises(ValueError, match="Unknown"):
        runner.route("both")


@pytest.mark.parametrize("damage", ["bytes", "record", "sample", "summary", "extra", "manifest"])
def test_shared_cache_rejects_corruption_and_missing_admission(tmp_path, damage):
    value, image = make_shared(tmp_path)
    item = next(shared.items_for(value))
    shared.verify_cache(tmp_path)
    if damage == "bytes": image.write_bytes(b"broken")
    elif damage == "record": protocol.atomic_json(tmp_path / "records/0000_0.json", {})
    elif damage == "sample":
        rec = protocol.read_json(tmp_path / "records/0000_0.json")
        rec["sample"]["anchor_seed"] += 1
        protocol.atomic_json(tmp_path / "records/0000_0.json", rec)
    elif damage == "summary": (tmp_path / "generation_summary.json").unlink()
    elif damage == "extra": Image.new("RGB", (1024, 640)).save(tmp_path / "images/extra.png")
    else:
        rec = protocol.read_json(tmp_path / "generation_summary.json")
        rec["image_manifest_sha256"] = "0"*64
        protocol.atomic_json(tmp_path / "generation_summary.json", rec)
    with pytest.raises((ValueError, OSError, KeyError)):
        shared.verify_cache(tmp_path)
    if damage in {"bytes", "record", "sample"}:
        assert not shared.cached_image(tmp_path, value, item)


def test_shared_worker_generates_once_and_reuses_without_loading_ssd(tmp_path, monkeypatch):
    value = shared_contract()
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    calls = []
    class Pipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["prompt"] == "panda native suffix"
            assert kwargs["generator"].initial_seed() == 2026
            return SimpleNamespace(images=[Image.new("RGB", (1024, 640), "red")])
    def load(*_):
        calls.append("load")
        return Pipeline()
    monkeypatch.setattr(shared, "load_native_ssd", load)
    args = SimpleNamespace(output_dir=tmp_path, rank=0, workers=1)
    shared.generation_worker(args, device=torch.device("cpu"))
    shared.summarize(tmp_path)
    shared.generation_worker(args, device=torch.device("cpu"))
    assert len(calls) == 2 and calls[0] == "load"
    receipt = protocol.read_json(tmp_path / "worker_receipts/rank_0.json")
    assert receipt["generated_images"] == 0 and receipt["reused_images"] == 1
    shared.verify_cache(tmp_path)


@pytest.mark.parametrize("mode", runner.MODES)
def test_real_mp4_pair_caches_route_and_shared_pixels(tmp_path, mode):
    frames, kw = make_pair(tmp_path, mode)
    assert runner.cached_pair(kw["output"], kw["contract"], kw["item"], kw["image_sha"])
    records = runner.pair_records(kw["output"], kw["contract"], kw["item"], kw["image_sha"], decode=True)
    assert records[0]["base_frames_sha256"] == records[1]["base_frames_sha256"]
    for rec in records:
        assert rec["system"] == runner.route(mode)["system"]
        assert rec["external_anchor"] == (mode == "ssd1b_anchor")
        assert rec["image_to_smolvlm2"] == (mode == "image_condition")
        assert rec["first_frame_generated"] == (mode == "image_condition")
    runner.verify_generation(kw["output"])
    assert protocol.read_json(kw["output"] / "generation_summary.json")["image_generation"] is False
    protocol.atomic_json(kw["output"] / "quicksr/scores/generation_snapshot.json", {"sha256":"old"})
    runner.verify_generation(kw["output"])
    assert len(list((kw["output"] / "quicksr").glob("retired_scores_*"))) == 1
    runner.verify_generation(kw["output"])
    assert len(list((kw["output"] / "quicksr").glob("retired_scores_*"))) == 1


@pytest.mark.parametrize("field", ["system", "external_anchor", "image_to_smolvlm2", "first_frame_generated",
                                    "shared_image_sha256", "base_frames_sha256", "conditioning_sha256", "unit_statistics"])
def test_wrong_metadata_invalidates_resume(tmp_path, field):
    _, kw = make_pair(tmp_path)
    path = kw["output"] / "quicksr/video_metadata/0000_0.json"
    rec = protocol.read_json(path)
    rec[field] = [{"supplied_anchor":False}] if field == "unit_statistics" else "invalid"
    protocol.atomic_json(path, rec)
    assert not runner.cached_pair(kw["output"], kw["contract"], kw["item"], kw["image_sha"])
    with pytest.raises(ValueError): runner.verify_generation(kw["output"])


def test_read_image_checks_exact_bytes_and_resize(tmp_path):
    _, path = make_shared(tmp_path)
    digest = protocol.sha256(path)
    assert runner.load_shared_image(path, digest).size == (512, 320)
    Image.new("RGB", (1024, 640), "blue").save(path)
    with pytest.raises(ValueError, match="changed before use"):
        runner.load_shared_image(path, digest)


@pytest.mark.parametrize("field", ["prompts", "max_prompts", "full_info_sha256", "samples_per_prompt", "seed",
                                    "seed_mapping", "config_sha256", "library_versions", "vbench_source_commit", "checkpoint"])
def test_mismatched_image_video_protocols_are_rejected(field):
    images = shared_contract()
    template = deepcopy(images)
    template["checkpoint"] = {"sha256":shared.FINAL_SHA256}
    runner.validate_shared_protocol(template, images)
    template[field] = {"sha256":"0"*64} if field == "checkpoint" else "wrong"
    with pytest.raises(ValueError): runner.validate_shared_protocol(template, images)


@pytest.mark.parametrize("mode", runner.MODES)
def test_worker_uses_trained_checkpoint_and_only_requested_image_path(tmp_path, monkeypatch, mode):
    from tools import infer_mobileov_stage1 as infer
    from new_mobile_ov.generation import quicksrnet

    _, image = make_shared(tmp_path / "shared")
    output = tmp_path / mode
    value = video_contract(tmp_path / "shared", mode)
    quicksr = tmp_path / "sr.pt"; quicksr.write_bytes(b"fake sr")
    value["quicksr"] = dict(path=str(quicksr), sha256=protocol.sha256(quicksr))
    protocol.atomic_json(output / "run_contract.json", value)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    calls = []
    def encoder(*args, **kwargs):
        calls.append(("condition", len(args), kwargs))
        return torch.zeros(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 4)
    def load(*args):
        calls.append("load trained")
        return encoder, lambda *v: v, None, None, None, dict(trained_dit_loaded=True, trained_connector_loaded=True)
    monkeypatch.setattr(base, "read_checkpoint", lambda *a, **kw: ({"step":283977}, shared.FINAL_SHA256, 283977))
    monkeypatch.setattr(base.stage2, "load_models", load)
    monkeypatch.setattr(infer, "autocast", lambda _: nullcontext())
    def encode(vae, video, generator, **kw):
        assert mode == "ssd1b_anchor"
        assert tuple(video.shape) == (1, 3, 1, 320, 512)
        assert generator.initial_seed() == 2026 + 228
        calls.append("VAE encode")
        return "latent"
    monkeypatch.setattr(infer, "encode_anchor", encode)
    def generate(*args, **kw):
        calls.append("generate")
        assert kw["anchor_latent"] == ("latent" if mode == "ssd1b_anchor" else None)
        assert (kw["seed"], kw["first_steps"], kw["video_steps"], kw["num_frames"]) == (2026, 20, 10, 49)
        assert (kw["guidance"], kw["video_guidance"]) == (7., 5.)
        return [Image.new("RGB", (32, 16), "red")]*3, [dict(unit=0, supplied_anchor=mode == "ssd1b_anchor")]
    monkeypatch.setattr(infer, "generate_frames", generate)
    monkeypatch.setattr(quicksrnet, "load_quicksrnet", lambda **kw: (torch.nn.Upsample(scale_factor=2), None))
    monkeypatch.setattr(shared, "load_native_ssd", lambda *_: pytest.fail("Evaluation must never load SSD1B"))
    args = SimpleNamespace(output_dir=output, rank=0, workers=1, mode=mode)
    runner.generation_worker(args, device=torch.device("cpu"))
    condition_calls = [c for c in calls if isinstance(c, tuple)]
    assert condition_calls[0] == ("condition", 1, {"drop_condition":True})
    assert condition_calls[1] == ("condition", 2 if mode == "image_condition" else 1,
                                  {"allow_text_image":True} if mode == "image_condition" else {})
    assert calls.count("generate") == 1 and calls.count("VAE encode") == (mode == "ssd1b_anchor")
    previous = list(calls)
    runner.generation_worker(args, device=torch.device("cpu"))
    assert calls == previous
    assert protocol.read_json(output / "worker_receipts/rank_0.json")["generated_pairs"] == 0


@pytest.mark.parametrize("mode", runner.MODES)
def test_score_assignments_and_report_include_both_variants_all_dimensions(tmp_path, monkeypatch, mode):
    value = dict(mode=mode, **runner.route(mode), max_prompts=0, training_complete=True, samples_per_prompt=5,
                 scope=f"{mode} shared SSD1B, raw/QuickSR")
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kw: commands.append(argv))
    for rank in range(8):
        runner.score_worker(SimpleNamespace(output_dir=tmp_path, rank=rank, workers=8, vbench_repo=tmp_path))
    assert len(commands) == 32
    assert len({c[c.index("--output")+1] for c in commands}) == 32
    assert all(c[c.index("--expected-samples-per-prompt")+1] == "5" for c in commands)
    score = dict(quality_score=.8, semantic_score=.6, total_score=.76, raw={d:.5 for d in runner.DIMENSIONS})
    module = ModuleType("evaluate_vbench_resume"); module.tabulate = lambda _: deepcopy(score)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner, "verify_generation", lambda _: None)
    protocol.atomic_json(tmp_path / "score_assets.json", [])
    for variant in base.VARIANTS:
        for dimension in runner.DIMENSIONS:
            protocol.atomic_json(tmp_path / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json",
                                 {dimension:[.5, []]})
    runner.summarize(SimpleNamespace(output_dir=tmp_path))
    report = protocol.read_json(tmp_path / "quicksr_ablation.json")
    assert set(report["scores"]) == set(base.VARIANTS)
    assert report["scope"] == value["scope"]
    assert all(d in (tmp_path / "quicksr_ablation.md").read_text() for d in runner.DIMENSIONS)
    value["max_prompts"] = 2
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    with pytest.raises(ValueError, match="No full score"): runner.summarize(SimpleNamespace(output_dir=tmp_path))


def test_cli_requires_final_checkpoint_and_separate_outputs():
    for mode in runner.MODES:
        args = runner.parse_args(["--mode", mode])
        assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
        assert args.expected_sha256 == shared.FINAL_SHA256
        assert args.checkpoint == base.DEFAULT_CHECKPOINT
    with pytest.raises(ValueError, match="completed epoch 3"):
        runner.parse_args(["--mode", "ssd1b_anchor", "--max-prompts", "1", "--allow-incomplete-checkpoint"])
    with pytest.raises(ValueError, match="separate"):
        runner.parse_args(["--mode", "image_condition", "--frames-root", "/tmp/same", "--output-dir", "/tmp/same"])
    for flags in (["--samples-per-prompt", "1"], ["--workers", "9"], ["--max-prompts", "-1"]):
        with pytest.raises(SystemExit): shared.parse_args(flags)


@pytest.mark.parametrize("step", [15000, 130000, 190000])
def test_shared_admission_rejects_any_earlier_checkpoint(tmp_path, monkeypatch, step):
    monkeypatch.setattr(base, "read_checkpoint", lambda *a, **kw: ({"step":step}, "sha", 283977))
    args = shared.parse_args(["--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="final epoch-3"):
        shared.prepare(args)


@pytest.mark.parametrize("mode", runner.MODES)
def test_preflight_forks_the_old_numeric_sampler_without_altering_it(tmp_path, monkeypatch, mode):
    images, _ = make_shared(tmp_path / "shared")
    template = deepcopy(images)
    template.update(checkpoint=dict(step=283977, sha256=shared.FINAL_SHA256), implementation_sha256={},
                    scope="old T2V", external_anchor=False, system=base.SYSTEM)
    original = deepcopy(template)
    args = SimpleNamespace(output_dir=tmp_path / mode, frames_root=tmp_path / "shared", mode=mode)
    def prepare(a):
        assert a.output_dir == args.output_dir / "_asset_preflight"
        protocol.atomic_json(a.output_dir / "vbench_info.json", [{"prompt_en":"panda"}])
        protocol.atomic_json(a.output_dir / "checkpoint_audit.json", {"step":283977})
        return template
    monkeypatch.setattr(base, "prepare", prepare)
    value = runner.prepare(args)
    assert template == original
    assert value["system"] == runner.route(mode)["system"]
    assert value["first_frame_generated"] == (mode == "image_condition")
    assert value["shared_ssd1b"]["path"] == str(args.frames_root)
    assert args.output_dir / "sample_plan.json" != args.frames_root / "sample_plan.json"
    assert set(value["implementation_sha256"]) == {"tools/benchmark_stage2_ssd1b_conditioning_dual.py",
                                                    "tools/prepare_vbench_stage2_ssd1b_frames.py"}


def test_changed_shared_manifest_cannot_be_used_by_eval(tmp_path):
    make_shared(tmp_path / "shared")
    value = video_contract(tmp_path / "shared", "image_condition")
    runner.checked_shared(value)
    protocol.atomic_json(tmp_path / "shared/image_manifest.json", [])
    with pytest.raises(ValueError, match="changed after video preflight"):
        runner.checked_shared(value)


@pytest.mark.parametrize("module", [shared, runner])
def test_full_gpu_run_cannot_start_on_a_login_node(monkeypatch, module):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(phase="run"))
    with pytest.raises(RuntimeError, match="sbatch"):
        module.main()


@pytest.mark.parametrize("script", ["prepare_vbench_stage2_ssd1b_frames.py", "benchmark_stage2_ssd1b_conditioning_dual.py"])
def test_cli_help_survives_foreign_tools_namespace(tmp_path, script):
    foreign = tmp_path / "tools"; foreign.mkdir(); (foreign / "__init__.py").write_text("FOREIGN=True\n")
    code = "import tools,runpy,sys; assert tools.FOREIGN; sys.argv=[sys.argv[1],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools" / script)], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH":str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--samples-per-prompt" in result.stdout


@pytest.fixture
def launch(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/install_vbench_env.sh").write_text('echo setup >> "$SETUP_LOG"\n')
    (root / "scripts/run_vbench_stage2_ssd1b_shared.sh").write_text((ROOT / "scripts/run_vbench_stage2_ssd1b_shared.sh").read_text())
    info = root / "checkpoints/vbench_repo/vbench/VBench_full_info.json"
    info.parent.mkdir(parents=True); info.write_text("[]")
    weight = root / "output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt"
    weight.parent.mkdir(parents=True); weight.write_bytes(b"final")
    shared_summary = root / "output/vbench_stage2_epoch3_ssd1b_conditioning_5x_dual/shared_ssd1b/generation_summary.json"
    shared_summary.parent.mkdir(parents=True); shared_summary.write_text("{}")
    binary = tmp_path / "bin"; binary.mkdir()
    for name, body in {"module":"exit 0", "srun":'while [[ "$1" == --* ]]; do shift; done; exec "$@"',
                       "squeue":'printf "%s\\n" "${ACTIVE_IDS:-}"',
                       "sbatch":'echo "$*" >> "$SBATCH_LOG"; N=$(wc -l < "$SBATCH_LOG"); echo "$((7000+N));cluster"'}.items():
        path = binary / name; path.write_text("#!/usr/bin/env bash\n" + body + "\n"); path.chmod(0o755)
    python = tmp_path / "env/bin/python"; python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
argv=sys.argv[1:]
pathlib.Path(os.environ['ARGV_LOG']).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index('--output-dir')+1]); output.mkdir(parents=True,exist_ok=True)
(output/'generation_summary.json').write_text('{}')
if '--mode' in argv:
    (output/'quicksr_ablation.json').write_text('{}')
    (output/'quicksr_ablation.md').write_text('mock raw/QuickSR report')
'''); python.chmod(0o755)
    remove = {"PROJECT_ROOT", "PAIR_ROOT", "ANCHOR_ROOT", "RUN_ROOT", "CHECKPOINT", "EXPECTED_SHA256", "STAGE2_PHASE",
              "MAX_PROMPTS", "SAMPLES_PER_PROMPT", "SEED", "CONFIG", "VBENCH_REPO", "HF_HOME", "TMPDIR"}
    env = {k:v for k,v in os.environ.items() if k not in remove}
    env.update(PATH=f"{binary}:{env['PATH']}", PROJECT_ROOT=str(root), SLURM_SUBMIT_DIR=str(root),
               CONDA_ENV=str(python.parents[1]), VBENCH_ENV=str(python.parents[1]), HF_HOME=str(tmp_path / "cache"),
               TMPDIR=str(tmp_path / "tmp"), ARGV_LOG=str(tmp_path / "argv.json"), SETUP_LOG=str(tmp_path / "setup.log"),
               SBATCH_LOG=str(tmp_path / "sbatch.log"))
    def run(script, **overrides):
        result = subprocess.run(["bash", str(ROOT / "scripts" / script)], env={**env, **overrides}, capture_output=True, text=True)
        log = Path(env["ARGV_LOG"])
        return result, json.loads(log.read_text()) if log.exists() else []
    return root, env, run, weight, shared_summary


@pytest.mark.parametrize("mode", WRAPPERS)
def test_berzelius_wrappers_use_same_images_final_weight_eight_gpus(launch, mode):
    root, env, run, _, _ = launch
    result, argv = run(WRAPPERS[mode])
    assert result.returncode == 0, result.stderr
    def option(name): return argv[argv.index(name)+1]
    assert option("--workers") == "8" and option("--samples-per-prompt") == "5"
    assert option("--expected-sha256") == shared.FINAL_SHA256
    assert option("--checkpoint").endswith("17641364/stage2_dit_connector_latest.pt")
    if mode == "frames":
        assert "tools/prepare_vbench_stage2_ssd1b_frames.py" in argv
        assert option("--output-dir").endswith("/shared_ssd1b")
    else:
        assert option("--mode") == mode and option("--frames-root").endswith("/shared_ssd1b")
        assert option("--output-dir").endswith("/"+mode)
        assert "mock raw/QuickSR report" in result.stdout
    assert "#SBATCH --gres=gpu:8" in (ROOT / "scripts" / WRAPPERS[mode]).read_text()


@pytest.mark.parametrize("failure", ["missing_weight", "missing_images", "bad_samples", "bad_seed"])
def test_launcher_failures_happen_before_setup(launch, failure):
    _, env, run, weight, summary = launch
    overrides = {}
    if failure == "missing_weight": weight.unlink()
    elif failure == "missing_images": summary.unlink()
    elif failure == "bad_samples": overrides["SAMPLES_PER_PROMPT"] = "1"
    else: overrides["SEED"] = "-1"
    result, argv = run(WRAPPERS["ssd1b_anchor"], **overrides)
    assert result.returncode != 0 and not argv
    assert not Path(env["SETUP_LOG"]).exists()


def test_submit_dependencies_and_duplicate_job_guard(launch):
    root, env, run, _, _ = launch
    script = "submit_vbench_stage2_ssd1b_conditioning_pair.sh"
    result, _ = run(script)
    assert result.returncode == 0, result.stderr
    lines = Path(env["SBATCH_LOG"]).read_text().splitlines()
    assert len(lines) == 3
    assert "prepare_vbench_stage2_ssd1b_frames" in lines[0]
    assert all("--dependency=afterok:7001 --kill-on-invalid-dep=yes" in line for line in lines[1:])
    assert "ssd1b_anchor" in lines[1] and "image_condition" in lines[2]
    result, _ = run(script, ACTIVE_IDS="7001\n7002\n7003")
    assert result.returncode != 0 and "already has active job" in result.stderr
    assert len(Path(env["SBATCH_LOG"]).read_text().splitlines()) == 3


@pytest.mark.parametrize("override", [{"SAMPLES_PER_PROMPT":"1"}, {"RUN_ROOT":"shared wrong root"}, {"CHECKPOINT":"missing.pt"}])
def test_submit_rejects_invalid_inputs_before_creating_jobs(launch, override):
    _, env, run, _, _ = launch
    result, _ = run("submit_vbench_stage2_ssd1b_conditioning_pair.sh", **override)
    assert result.returncode != 0 and not Path(env["SBATCH_LOG"]).exists()


def test_all_new_shell_files_parse():
    names = list(WRAPPERS.values()) + ["run_vbench_stage2_ssd1b_shared.sh", "submit_vbench_stage2_ssd1b_conditioning_pair.sh",
                                      "test_vbench_stage2_ssd1b_conditioning_local.sbatch"]
    for name in names:
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)
