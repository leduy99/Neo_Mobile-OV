from contextlib import nullcontext
from copy import deepcopy
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

from tools import benchmark_mobileov_stage2_media3epochs_dual as runner
from tools import vbench_joint_native_protocol as protocol
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/vbench_mobileov_stage2_media3epochs_1node8gpu.sbatch"


def payload():
    return dict(format=runner.stage2.FORMAT, step=12, connector_spec={"width": 2},
                connector={"weight": torch.ones(2, 2)}, dit={"weight": torch.ones(2, 2)},
                contract=dict(config_sha256="cfg", processor="processor", processor_sha256="p", max_tokens=512,
                              smolvlm2_sha256="smol", initial_weights_sha256={},
                              tasks=["t2i", "t2v", "i2v_reference", "i2v_anchor", "image_reconstruction"],
                              media_epochs=dict(recipe="media_epochs_i2v_v1", epochs=3, steps=12, steps_per_epoch=4)))


def test_final_checkpoint_required_and_old_130k_cannot_pass_as_three_epochs():
    value = payload()
    assert runner.check_media_checkpoint(value) == 12
    value["step"] = 5
    with pytest.raises(ValueError, match="NOT complete"):
        runner.check_media_checkpoint(value)
    assert runner.check_media_checkpoint(value, allow_incomplete=True) == 12
    value["step"] = 130000
    value["contract"]["media_epochs"].update(steps=283977, steps_per_epoch=94659)
    with pytest.raises(ValueError, match="130000.*283977"):
        runner.check_media_checkpoint(value)


@pytest.mark.parametrize("change", ["format", "recipe", "epochs", "budget", "tasks", "nan"])
def test_other_checkpoints_and_invalid_weights_are_rejected(change):
    value = payload()
    if change == "format": value["format"] = "joint_v2"
    elif change == "recipe": value["contract"]["media_epochs"]["recipe"] = "other"
    elif change == "epochs": value["contract"]["media_epochs"]["epochs"] = 2
    elif change == "budget": value["contract"]["media_epochs"]["steps_per_epoch"] = 5
    elif change == "tasks": value["contract"]["tasks"] = ["t2v"]
    else: value["dit"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError): runner.check_media_checkpoint(value)


def test_missing_checkpoint_never_falls_back_to_old_hf(tmp_path):
    with pytest.raises(FileNotFoundError, match="No old HF fallback"):
        runner.read_checkpoint(tmp_path / "missing.pt")
    path = tmp_path / "weights.pt"
    torch.save(payload(), path)
    _, digest, target = runner.read_checkpoint(path)
    assert target == 12 and digest == protocol.sha256(path)
    with pytest.raises(ValueError, match="SHA256 changed"):
        runner.read_checkpoint(path, expected_sha="0" * 64)


def test_cli_defaults_are_eight_gpus_five_samples_and_remote_output_weights():
    args = runner.parse_args([])
    assert args.workers == 8 and args.samples_per_prompt == 5 and args.max_prompts == 0
    assert "/output/mobileov_stage2_media3epochs/17641364/" in str(args.checkpoint)
    assert not args.allow_incomplete_checkpoint
    for flags in (["--samples-per-prompt", "1"], ["--allow-incomplete-checkpoint"], ["--workers", "9"]):
        with pytest.raises(SystemExit): runner.parse_args(flags)
    args = runner.parse_args(["--max-prompts", "2", "--samples-per-prompt", "1", "--allow-incomplete-checkpoint"])
    assert args.allow_incomplete_checkpoint


def contract():
    return dict(format="mobileov_stage2_media3epochs_vbench_dual_v1", prompts=["panda"], samples_per_prompt=1,
                seed=2026, max_prompts=1, training_complete=False,
                checkpoint=dict(path="unused", sha256="unused"), config="unused", implementation_sha256={},
                quicksr=dict(path="unused"),
                exports={"no_quicksr":dict(width=32,height=16,frames=3,fps=24),
                         "quicksr":dict(width=64,height=32,frames=3,fps=24)})


def fixture_pair(tmp_path):
    value = contract()
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    item = next(runner.items_for(value))
    array = np.random.default_rng(7).integers(0, 256, (3, 16, 32, 3), dtype=np.uint8)
    frames = [Image.fromarray(frame) for frame in array]
    kwargs = dict(output=tmp_path, contract=value, item=item, model=torch.nn.Upsample(scale_factor=2, mode="nearest"),
                  device=torch.device("cpu"), statistics=[dict(unit=0, generated=True)])
    return frames, kwargs


def test_global_sample_plan_has_no_anchor_and_disjoint_8_gpu_shards():
    value = dict(prompts=[f"prompt {i}" for i in range(944)], samples_per_prompt=5, seed=2026)
    items = list(runner.items_for(value))
    assert len(items) == len({item["seed"] for item in items}) == 4720
    assert items[0]["seed"] == 2026 and items[5]["seed"] == 2027
    assert all("anchor_seed" not in item for item in items)
    shards = [protocol.shard_items(items, rank, 8) for rank in range(8)]
    assert len({item["key"] for shard in shards for item in shard}) == 4720


def test_real_dual_mp4_exports_decode_and_resume_without_external_anchor(tmp_path):
    frames, kwargs = fixture_pair(tmp_path)
    runner.export_pair(frames, **kwargs)
    assert runner.cached_pair(tmp_path, kwargs["contract"], kwargs["item"])
    assert not list(tmp_path.glob("**/anchors"))
    for variant in runner.VARIANTS:
        record = protocol.read_json(tmp_path / variant / "video_metadata/0000_0.json")
        assert record["external_anchor"] is False and record["first_frame_generated"] is True
        assert protocol.decode_video(tmp_path / variant / "videos/panda-0.mp4", kwargs["contract"]["exports"][variant])
    runner.verify_generation(tmp_path)
    assert protocol.read_json(tmp_path / "generation_summary.json")["same_generated_frames"] is True


def test_wrong_pair_hash_or_corrupt_export_invalidates_cache(tmp_path):
    frames, kwargs = fixture_pair(tmp_path)
    runner.export_pair(frames, **kwargs)
    path = tmp_path / "no_quicksr/video_metadata/0000_0.json"
    record = protocol.read_json(path)
    record["base_frames_sha256"] = "0" * 64
    protocol.atomic_json(path, record)
    assert not runner.cached_pair(tmp_path, kwargs["contract"], kwargs["item"])
    with pytest.raises(ValueError, match="share generated frames"):
        runner.verify_generation(tmp_path)
    runner.export_pair(frames, **kwargs)
    (tmp_path / "quicksr/videos/panda-0.mp4").write_bytes(b"broken")
    assert not runner.cached_pair(tmp_path, kwargs["contract"], kwargs["item"])


def test_changed_video_hash_retires_cached_scores(tmp_path):
    frames, kwargs = fixture_pair(tmp_path)
    runner.export_pair(frames, **kwargs)
    protocol.atomic_json(tmp_path / "quicksr/scores/generation_snapshot.json", dict(sha256="old"))
    runner.verify_generation(tmp_path)
    assert len(list((tmp_path / "quicksr").glob("retired_scores_*"))) == 1
    runner.verify_generation(tmp_path)
    assert len(list((tmp_path / "quicksr").glob("retired_scores_*"))) == 1


def test_worker_loads_trained_stage2_stack_and_generates_each_clip_once(tmp_path, monkeypatch):
    from tools import infer_mobileov_stage1 as infer
    from new_mobile_ov.generation import quicksrnet

    value = contract()
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-unit-test")
    monkeypatch.setattr(runner, "read_checkpoint", lambda *args, **kw: (payload(), "sha", 12))
    calls = []

    def encoder(prompt, **kwargs):
        return (torch.zeros(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 4))

    def load_models(args, loaded, device):
        calls.append("load")
        assert loaded["format"] == runner.stage2.FORMAT
        return encoder, lambda *parts: parts, None, None, None, dict(trained_dit_loaded=True, trained_connector_loaded=True)

    monkeypatch.setattr(runner.stage2, "load_models", load_models)
    monkeypatch.setattr(infer, "autocast", lambda _: nullcontext())
    def generate(*args, **kw):
        calls.append("generate")
        assert "anchor_latent" not in kw and kw["num_frames"] == 49
        assert (kw["first_steps"], kw["video_steps"], kw["guidance"], kw["video_guidance"]) == (20,10,7,5)
        return [Image.new("RGB", (32,16), "red")] * 3, [dict(unit=0, generated=True)]
    monkeypatch.setattr(infer, "generate_frames", generate)
    monkeypatch.setattr(quicksrnet, "load_quicksrnet", lambda **kw: (torch.nn.Upsample(scale_factor=2), None))
    args = SimpleNamespace(output_dir=tmp_path, rank=0, workers=1)
    runner.generation_worker(args, device=torch.device("cpu"))
    assert calls == ["load", "generate"]
    runner.generation_worker(args, device=torch.device("cpu"))
    assert calls == ["load", "generate"]
    assert protocol.read_json(tmp_path / "weight_audits/rank_0.json")["trained_connector_loaded"] is True


def test_two_variants_cover_all_32_vbench_tasks_and_keep_paths_separate(tmp_path, monkeypatch):
    protocol.atomic_json(tmp_path / "run_contract.json", dict(samples_per_prompt=5))
    tasks = [task for group in runner.dual.score_assignments(8, [runner.SYSTEM]) for task in group]
    assert len(tasks) == len(set(tasks)) == 32
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kw: commands.append(argv))
    for rank in range(8):
        runner.score_worker(SimpleNamespace(output_dir=tmp_path, vbench_repo=tmp_path, rank=rank, workers=8))
    assert len(commands) == 32
    for argv in commands:
        videos = Path(option(argv, "--videos")); scores = Path(option(argv, "--output"))
        assert videos.parent == scores.parents[1]
        assert option(argv, "--expected-samples-per-prompt") == "5"


def test_full_summary_writes_all_dimensions_and_rejects_incomplete_smoke(tmp_path, monkeypatch):
    value = contract()
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    with pytest.raises(ValueError, match="No full Stage-2 score"):
        runner.summarize(SimpleNamespace(output_dir=tmp_path))
    value.update(max_prompts=0, training_complete=True, scope="Stage 2 T2V only")
    protocol.atomic_json(tmp_path / "run_contract.json", value)
    protocol.atomic_json(tmp_path / "score_assets.json", [])
    score = dict(quality_score=.8,semantic_score=.6,total_score=.76,raw={d:.5 for d in runner.DIMENSIONS})
    module = ModuleType("evaluate_vbench_resume"); module.tabulate = lambda _: deepcopy(score)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    for variant in runner.VARIANTS:
        for dimension in runner.DIMENSIONS:
            protocol.atomic_json(tmp_path / variant / "scores" / dimension / "vbench" / f"{dimension}_eval_results.json",
                                 {dimension:[.5,[]]})
    runner.summarize(SimpleNamespace(output_dir=tmp_path))
    report = protocol.read_json(tmp_path / "quicksr_ablation.json")
    assert report["scope"] == value["scope"]
    assert set(report["scores"]["quicksr"][runner.SYSTEM]["raw"]) == set(runner.DIMENSIONS)
    assert "no external anchor" in (tmp_path / "quicksr_ablation.md").read_text()


def test_berzelius_launcher_uses_final_local_checkpoint_and_no_hf_download(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, _, run = launch
    path = root / "output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt"
    path.parent.mkdir(parents=True); path.write_bytes(b"local final weights")
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert "tools/benchmark_mobileov_stage2_media3epochs_dual.py" in argv
    assert option(argv, "--checkpoint") == "output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt"
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert "--allow-incomplete-checkpoint" not in argv and "--hf-file" not in argv
    assert "mock report" in result.stdout
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "#SBATCH --gres=gpu:8" in SCRIPT.read_text()


def test_cli_handles_already_imported_foreign_tools_package(tmp_path):
    foreign = tmp_path / "tools"; foreign.mkdir(); (foreign / "__init__.py").write_text("FOREIGN = True\n")
    script = ROOT / "tools/benchmark_mobileov_stage2_media3epochs_dual.py"
    code = "import tools,runpy,sys; assert tools.FOREIGN; sys.argv=[sys.argv[1],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable,"-c",code,str(script)], cwd=tmp_path,
                            env={**os.environ,"PYTHONPATH":str(tmp_path)}, capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert "--allow-incomplete-checkpoint" in result.stdout
