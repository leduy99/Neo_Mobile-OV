from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import benchmark_joint_v2_native as runner
from tools import generate_vbench_joint_v2_native as worker
from tools import vbench_joint_native_protocol as protocol


def test_defaults_are_full_paired_5_seed_8_gpu():
    args = runner.parse_args([])
    assert args.samples_per_prompt == 5 and args.workers == 8 and args.max_prompts == 0
    assert "17346642" in str(args.joint_checkpoint)
    assert "17291461" in str(args.image_checkpoint)
    assert protocol.SYSTEMS == ("joint_v2", "native_hybrid")


@pytest.mark.parametrize("flags", [["--workers", "0"], ["--workers", "9"], ["--rank", "8"],
                                  ["--samples-per-prompt", "0"], ["--max-prompts", "-1"], ["--seed", "-1"]])
def test_bad_configuration_rejected(flags):
    with pytest.raises(SystemExit):
        runner.parse_args(flags)


def test_full_standard_requires_5_seeds_but_smoke_can_use_one():
    with pytest.raises(SystemExit):
        runner.parse_args(["--samples-per-prompt", "1"])
    args = runner.parse_args(["--samples-per-prompt", "1", "--max-prompts", "2"])
    assert args.samples_per_prompt == 1 and args.max_prompts == 2


def test_trained_only_full_1x_is_an_explicit_protocol():
    args = runner.parse_args(["--systems", "joint_v2", "--protocol", "paired1x", "--workers", "2"])
    assert args.systems == ["joint_v2"] and args.samples_per_prompt == 1 and args.max_prompts == 0
    with pytest.raises(SystemExit):
        runner.parse_args(["--protocol", "paired1x", "--samples-per-prompt", "5"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--systems", "joint_v2", "joint_v2"])


def test_selected_model_has_only_16_scoring_tasks_on_2_gpus():
    tasks = [task for group in protocol.score_assignments(2, ["joint_v2"]) for task in group]
    assert len(tasks) == len(set(tasks)) == 16
    assert {system for system, _ in tasks} == {"joint_v2"}


def test_prompt_normalization_preserves_dimension_memberships(tmp_path):
    path = tmp_path / "info.json"
    protocol.atomic_json(path, [dict(prompt_en="  A  panda  ", dimension=["scene"]),
                               dict(prompt_en="A panda", dimension=["object_class"]),
                               dict(prompt_en="A car", dimension=["color"])])
    prompts, rows = protocol.load_prompt_rows(path, 1)
    assert prompts == ["A panda"] and len(rows) == 2
    assert {row["dimension"][0] for row in rows} == {"scene", "object_class"}


@pytest.mark.parametrize("value", [None, "", "../escape", "a/b", "a\\b"])
def test_unsafe_prompts_rejected(tmp_path, value):
    path = tmp_path / "info.json"
    protocol.atomic_json(path, [dict(prompt_en=value)])
    with pytest.raises(ValueError):
        protocol.load_prompt_rows(path)


def test_global_seeds_and_complete_disjoint_shards():
    items = list(protocol.sample_items([f"prompt {i}" for i in range(944)], 5, 2026))
    assert len(items) == 4720
    assert len({item["video_seed"] for item in items}) == 4720
    for workers in (1, 3, 8):
        shards = [protocol.shard_items(items, rank, workers) for rank in range(workers)]
        flat = [item for shard in shards for item in shard]
        assert len(flat) == len(items) and len({item["key"] for item in flat}) == len(items)
        assert sorted(flat, key=lambda item: item["video_seed"]) == items
    assert items[5]["filename"] == "prompt 1-0.mp4"
    assert items[5]["anchor_seed"] == items[5]["video_seed"] == 2031


def test_long_filename_rejected_before_loading_models():
    with pytest.raises(ValueError, match="filesystem limit"):
        list(protocol.sample_items(["a" * 255], 5, 0))


def test_temporary_movie_name_does_not_extend_a_long_prompt(tmp_path):
    item = list(protocol.sample_items(["a" * 248], 5, 0))[0]
    destination = tmp_path / item["filename"]
    assert len(destination.name.encode()) <= 255
    temporary = worker.temporary_video_path(destination, item)
    assert len(temporary.name.encode()) < 64
    assert temporary.parent == destination.parent and temporary.suffix == ".mp4"


def test_score_assignment_covers_32_tasks_once():
    groups = protocol.score_assignments(8)
    tasks = [task for group in groups for task in group]
    assert all(groups)
    assert len(tasks) == len(set(tasks)) == 32
    assert set(tasks) == {(system, i) for system in protocol.SYSTEMS for i in range(16)}
    loads = [sum(protocol.SCORE_COSTS[i] for _, i in group) for group in groups]
    assert max(loads) - min(loads) < 10


def test_contract_refuses_stale_settings(tmp_path):
    contract = dict(seed=2026, samples=5, quicksr=2)
    first = protocol.bind_contract(tmp_path, contract)
    assert protocol.bind_contract(tmp_path, contract) == first
    with pytest.raises(ValueError, match="Resume contract differs"):
        protocol.bind_contract(tmp_path, {**contract, "seed": 2027})


def checkpoint_payload(kind="joint"):
    value = dict(step=10000, teacher_stack=dict(name="multistep"),
                 bridge={"weight": torch.ones(2)}, dit={"weight": torch.ones(2)})
    if kind == "image":
        value.update(step=160000, config=dict(training_version="v11_balanced"))
    return value


@pytest.mark.parametrize("kind", ["joint", "image"])
def test_checkpoint_metadata_and_sha_checked_without_overwrite(tmp_path, kind):
    path = tmp_path / "model.pt"
    torch.save(checkpoint_payload(kind), path)
    original = protocol.sha256(path)
    record = protocol.ensure_checkpoint(path, "unused", original, kind)
    assert record["sha256"] == original
    with pytest.raises(ValueError, match="SHA mismatch"):
        protocol.ensure_checkpoint(path, "unused", "0" * 64, kind)
    assert protocol.sha256(path) == original
    wrong = checkpoint_payload(kind)
    wrong["step"] = 42
    torch.save(wrong, path)
    with pytest.raises(ValueError, match="Expected"):
        protocol.ensure_checkpoint(path, "unused", protocol.sha256(path), kind)
    assert path.is_file()


def test_download_is_pinned_and_validated_before_install(tmp_path, monkeypatch):
    import huggingface_hub

    source = tmp_path / "hub.pt"
    torch.save(checkpoint_payload(), source)
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: (calls.append((a, kw)) or str(source)))
    target = tmp_path / "new/model.pt"
    with pytest.raises(ValueError, match="Downloaded checkpoint SHA mismatch"):
        protocol.ensure_checkpoint(target, "joint.pt", "0" * 64, "joint")
    assert not target.exists() and not list(target.parent.glob("*.tmp.*"))
    protocol.ensure_checkpoint(target, "joint.pt", protocol.sha256(source), "joint")
    assert target.exists() and calls[-1][1]["revision"] == protocol.HF_REVISION


def test_decode_and_resume_reject_wrong_video_metadata(tmp_path):
    import cv2
    import numpy as np

    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 16))
    for i in range(3):
        writer.write(np.full((16, 32, 3), i * 50, dtype=np.uint8))
    writer.release()
    shape = dict(width=32, height=16, fps=24, frames=3)
    assert protocol.decode_video(video, shape)
    assert not protocol.decode_video(video, {**shape, "fps": 25})
    item = list(protocol.sample_items(["panda"], 1, 10))[0]
    record = tmp_path / "metadata.json"
    protocol.atomic_json(record, dict(sample=item, system="joint_v2", contract_sha256="abc",
                                     shape=shape, video_sha256=protocol.sha256(video)))
    assert protocol.cached_video(video, record, item, "joint_v2", "abc", shape)
    assert not protocol.cached_video(video, record, item, "native_hybrid", "abc", shape)
    assert not protocol.cached_video(video, record, item, "joint_v2", "other", shape)


def test_8_cpu_processes_get_disjoint_gpu_visibility(tmp_path):
    # Exercises orchestration without allocating or running any GPU kernels.
    commands = [[sys.executable, "-c", "import os,pathlib; pathlib.Path(os.environ['CUDA_VISIBLE_DEVICES']+'.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])"] for _ in range(8)]
    devices = [f"GPU-{i}" for i in range(8)]
    runner.run_processes(commands, devices, tmp_path / "logs", cwd=tmp_path)
    assert {p.read_text() for p in tmp_path.glob("GPU-*.txt")} == set(devices)


def test_one_worker_failure_stops_other_workers(tmp_path):
    commands = [[sys.executable, "-c", "raise SystemExit(3)"],
                [sys.executable, "-c", "import time; time.sleep(60)"]]
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="Worker failed"):
        runner.run_processes(commands, ["0", "1"], tmp_path / "logs", cwd=tmp_path)
    assert time.monotonic() - start < 10


def test_native_and_joint_generation_use_intended_conditioners(monkeypatch):
    module = ModuleType("neodragon.utils.generation_utils")
    module.DEFAULT_NEGATIVE_PROMPT = "negative"
    module.DEFAULT_PROMPT_MODIFIER = ", suffix"
    module.generate = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", module)
    native_text, native_context = torch.nn.Identity(), torch.nn.Identity()
    pipeline = SimpleNamespace(mode="hybrid", text_encoder_bundle=native_text, context_adapter=native_context,
        dit="dit", vae="vae", scheduler="scheduler", config=SimpleNamespace(frames_per_unit=1,
        stages=[1, 2, 4], gen_confs=dict(hybrid=dict(do_classifier_free_guidance=False))))
    native = worker.generate_frames(pipeline, None, None, "panda", "anchor", "native_hybrid", torch.device("cpu"), torch.float32)
    assert native["text_encoder_bundle"] is native_text and native["context_adapter"] is native_context
    assert native["image"] == "anchor" and native["num_frames"] == 49
    positive = (torch.ones(1, 2, 3), torch.ones(1, 2), torch.ones(1, 3))
    negative = tuple(torch.zeros_like(value) for value in positive)
    bridge = SimpleNamespace(encode=lambda prompts: (positive if prompts == ["panda, suffix"] else pytest.fail("wrong modifier")))
    pipeline.mode = "monolithic"
    pipeline.config.gen_confs["monolithic"] = dict(do_classifier_free_guidance=True, video_guidance_scale=5.)
    joint = worker.generate_frames(pipeline, bridge, negative, "panda", "anchor", "joint_v2", torch.device("cpu"), torch.float32)
    assert isinstance(joint["context_adapter"], torch.nn.Identity)
    assert joint["text_encoder_bundle"] is not native_text
    assert torch.equal(joint["text_encoder_bundle"](["negative"], torch.device("cpu"))[0], negative[0])
    assert joint["video_guidance_scale"] == 5.


def test_generation_hash_change_retires_scores(tmp_path):
    contract = dict(prompts=["panda"], samples_per_prompt=1, seed=1, output=dict(width=1))
    contract_sha = protocol.fingerprint(contract)
    item = list(protocol.sample_items(contract["prompts"], 1, 1))[0]
    for system in protocol.SYSTEMS:
        folder = tmp_path / system
        (folder / "videos").mkdir(parents=True)
        (folder / "anchors").mkdir()
        video, anchor = folder / "videos" / item["filename"], folder / "anchors" / f"{item['key']}.png"
        video.write_bytes(b"video")
        anchor.write_bytes(b"anchor")
        protocol.atomic_json(folder / "records" / f"{item['key']}.json", dict(
            sample=item, system=system, contract_sha256=contract_sha, shape=contract["output"],
            video_sha256=protocol.sha256(video), anchor_sha256=protocol.sha256(anchor)))
        protocol.atomic_json(folder / "scores/generation_snapshot.json", dict(sha256="stale"))
    runner.verify_generation(tmp_path, contract)
    assert len(list(tmp_path.glob("*/retired_scores_*"))) == 2
    runner.verify_generation(tmp_path, contract)
    assert len(list(tmp_path.glob("*/retired_scores_*"))) == 2


def test_joint_only_coverage_does_not_require_any_native_result(tmp_path):
    contract = dict(prompts=["panda"], samples_per_prompt=1, seed=1, output=dict(width=1),
                    selected_systems=["joint_v2"])
    item = list(protocol.sample_items(contract["prompts"], 1, 1))[0]
    folder = tmp_path / "joint_v2"
    (folder / "videos").mkdir(parents=True)
    (folder / "anchors").mkdir()
    video, anchor = folder / "videos" / item["filename"], folder / "anchors" / f"{item['key']}.png"
    video.write_bytes(b"video")
    anchor.write_bytes(b"anchor")
    protocol.atomic_json(folder / "records" / f"{item['key']}.json", dict(
        sample=item, system="joint_v2", contract_sha256=protocol.fingerprint(contract), shape=contract["output"],
        video_sha256=protocol.sha256(video), anchor_sha256=protocol.sha256(anchor)))
    runner.verify_generation(tmp_path, contract)
    assert protocol.read_json(tmp_path / "generation_summary.json")["systems"] == ["joint_v2"]
    assert not (tmp_path / "native_hybrid").exists()


def test_joint_only_assets_do_not_require_native_ssd_or_text_weights(tmp_path):
    for name in ("causal_video_vae", "diffusion_transformer_320p_multistep_t2v"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    (tmp_path / "diffusion_transformer_320p_multistep_t2v/unused_release.safetensors").write_bytes(b"not used")
    args = SimpleNamespace(native_model_dir=tmp_path, systems=["joint_v2"])
    _, records = runner.native_snapshot(args)
    assert len(records) == 2 and all(record["file"].endswith("config.json") for record in records)
    args.systems = ["native_hybrid"]
    with pytest.raises(ValueError, match="incomplete"):
        runner.native_snapshot(args)


def test_single_system_summary_never_claims_a_native_comparison(tmp_path, monkeypatch):
    scores = dict(quality_score=.8, semantic_score=.6, total_score=.76,
                  raw={dimension: .5 for dimension in protocol.DIMENSIONS})
    fake = ModuleType("evaluate_vbench_resume")
    fake.tabulate = lambda _: scores
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", fake)
    monkeypatch.setattr(runner, "verify_generation", lambda *_: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_kw: None)
    protocol.atomic_json(tmp_path / "run_contract.json", dict(selected_systems=["joint_v2"], protocol_name="paired1x"))
    protocol.atomic_json(tmp_path / "score_assets.json", [])
    for dimension in protocol.DIMENSIONS:
        protocol.atomic_json(tmp_path / "joint_v2/scores" / dimension / "vbench" / f"{dimension}_eval_results.json",
                             {dimension: [.5, []]})
    runner.summarize(SimpleNamespace(output_dir=tmp_path))
    report = protocol.read_json(tmp_path / "result.json")
    assert report["system"] == "joint_v2" and report["native_reference_evaluated"] is False
    assert not (tmp_path / "comparison.json").exists() and not (tmp_path / "native_hybrid").exists()


def test_aggregate_requires_both_systems_and_all_dimensions():
    scores = dict(quality_score=.8, semantic_score=.6, total_score=.76,
                  raw={dimension: .5 for dimension in protocol.DIMENSIONS})
    summaries = {system: deepcopy(scores) for system in protocol.SYSTEMS}
    assert protocol.aggregate_scores(summaries, {})["joint_minus_native_percentage_points"]["total_score"] == 0
    del summaries["native_hybrid"]["raw"]["scene"]
    with pytest.raises(ValueError, match="Incomplete dimensions"):
        protocol.aggregate_scores(summaries, {})


def test_scoring_uses_separate_dimension_output_trees(tmp_path, monkeypatch):
    protocol.atomic_json(tmp_path / "run_contract.json", dict(samples_per_prompt=5))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    args = SimpleNamespace(output_dir=tmp_path, workers=8, rank=0, vbench_repo=tmp_path)
    runner.score_worker(args)
    for command in calls:
        assert command[command.index("--expected-samples-per-prompt") + 1] == "5"
        index = int(command[command.index("--start") + 1])
        destination = Path(command[command.index("--output") + 1])
        assert destination.name == protocol.DIMENSIONS[index]
        assert int(command[command.index("--end") + 1]) == index + 1


def test_evaluation_cache_is_prepared_once_and_changes_are_rejected(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    weight, linear = cache / "weight.pt", cache / "linear/sa_0_4_vit_l_14_linear.pth"
    weight.write_bytes(b"weight")
    linear.parent.mkdir()
    linear.write_bytes(b"predictor")
    utils, aesthetic = ModuleType("vbench.utils"), ModuleType("vbench.aesthetic_quality")
    utils.CACHE_DIR = str(cache)
    calls = []
    utils.init_submodules = lambda dimensions, **kwargs: (calls.append(tuple(dimensions)) or
        dict(subject_consistency=dict(path=str(weight)), aesthetic_quality=[str(weight), str(linear.parent)]))
    aesthetic.get_aesthetic_model = lambda _: None
    monkeypatch.setitem(sys.modules, "vbench.utils", utils)
    monkeypatch.setitem(sys.modules, "vbench.aesthetic_quality", aesthetic)
    backend = SimpleNamespace(__file__=str(tmp_path / "complete_vbench/__init__.py"))
    monkeypatch.setattr(runner.importlib, "import_module", lambda _: backend)
    args = SimpleNamespace(output_dir=tmp_path / "results")
    runner.warm_scores(args)
    assert calls == [protocol.DIMENSIONS]
    assert len(protocol.read_json(args.output_dir / "score_assets.json")) == 2
    runner.warm_scores(args)
    weight.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="assets changed"):
        runner.warm_scores(args)


def test_incomplete_vbench_backend_is_rejected_before_downloads(tmp_path, monkeypatch):
    backend = SimpleNamespace(__file__="/partial/vbench/__init__.py")

    def import_module(name):
        if name == "vbench.aesthetic_quality":
            raise ModuleNotFoundError(name)
        return backend

    monkeypatch.setattr(runner.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="Incomplete/incompatible VBench backend.*aesthetic_quality"):
        runner.warm_scores(SimpleNamespace(output_dir=tmp_path))


def test_sbatch_is_8_gpu_and_full_5_seed_by_default():
    script = Path(__file__).resolve().parents[1] / "scripts/vbench_joint_v2_vs_native_quicksr_1node8gpu.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "#SBATCH --gres=gpu:8" in text and "--workers 8" in text
    assert '${SAMPLES_PER_PROMPT:-5}' in text and "NATIVE_MODEL_DIR" in text


def test_local_job_selects_only_joint_v2_1x_on_2_gpus():
    script = Path(__file__).resolve().parents[1] / "scripts/vbench_joint_v2_quicksr_local_2gpu.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for value in ("#SBATCH --gres=gpu:2", "--workers 2", "--systems joint_v2", "--protocol paired1x", "--samples-per-prompt 1"):
        assert value in text
    assert "--systems native_hybrid" not in text


@pytest.mark.parametrize("name", ["benchmark_joint_v2_native.py", "generate_vbench_joint_v2_native.py",
                                  "benchmark_joint_v2_quicksr_ablation.py", "generate_vbench_joint_v2_dual_export.py",
                                  "reproduce_native_neodragon_vbench.py", "generate_native_neodragon_release.py"])
def test_cli_imports_with_a_foreign_installed_tools_package(tmp_path, name):
    foreign = tmp_path / "tools"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("FOREIGN_PACKAGE = True\n")
    script = Path(__file__).resolve().parents[1] / "tools" / name
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    code = "import runpy,sys,tools; assert tools.FOREIGN_PACKAGE; sys.argv=[sys.argv[1],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
