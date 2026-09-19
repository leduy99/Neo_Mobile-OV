from contextlib import nullcontext
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tools import generate_vbench_mobileov_stage2 as run


def payload():
    return dict(format=run.FORMAT, step=100000, connector_spec={"width": 2},
                connector={"weight": torch.ones(2, 2)}, dit={"weight": torch.ones(2, 2)},
                contract=dict(config_sha256="config", processor="processor", processor_sha256="p",
                              max_tokens=2048, smolvlm2_sha256="s", initial_weights_sha256={}))


def test_defaults_are_current_stage2_without_external_anchor(tmp_path):
    args = run.parse_args(["--vbench-info", str(tmp_path / "info.json"), "--output-dir", str(tmp_path)])
    assert args.expected_step == 100000
    assert "17519025" in str(args.checkpoint)
    assert (args.frames, args.height, args.width, args.fps) == (49, 320, 512, 24)
    assert (args.first_steps, args.video_steps, args.guidance, args.video_guidance) == (20, 10, 7, 5)
    assert args.samples_per_prompt == 1 and args.max_prompts == 0
    assert not hasattr(args, "image_checkpoint")


@pytest.mark.parametrize("flags", [
    ["--samples-per-prompt", "0"], ["--samples-per-prompt", "6"],
    ["--height", "321"], ["--max-prompts", "-1"],
    ["--guidance", "nan"], ["--video-guidance", "0"], ["--frames", "50"],
])
def test_invalid_sampling_rejected(flags):
    with pytest.raises(SystemExit):
        run.parse_args(["--vbench-info", "info.json", "--output-dir", "out", *flags])


def test_prompt_dedup_order_and_seed_mapping(tmp_path):
    path = tmp_path / "info.json"
    run.atomic_json(path, [dict(prompt_en="  A  panda "), dict(prompt_en="A panda"), dict(prompt_en="A car")])
    prompts = run.load_prompts(path)
    assert prompts == ["A panda", "A car"]
    items = list(run.sample_items(prompts, 2, 11))
    assert [item["seed"] for item in items] == [11, 12, 13, 14]
    assert [item["filename"] for item in items] == ["A panda-0.mp4", "A panda-1.mp4", "A car-0.mp4", "A car-1.mp4"]


@pytest.mark.parametrize("prompt", ["../escape", "a/b", "a\\b", "", None])
def test_prompt_must_be_usable_by_official_filename_lookup(tmp_path, prompt):
    path = tmp_path / "info.json"
    run.atomic_json(path, [dict(prompt_en=prompt)])
    with pytest.raises(ValueError):
        run.load_prompts(path)


def test_long_filename_fails_before_model_loading():
    with pytest.raises(ValueError, match="filesystem limit"):
        list(run.sample_items(["a" * 255], 1, 0))


@pytest.mark.parametrize("change", ["phase", "step", "nan", "missing_dit", "contract"])
def test_checkpoint_validation(change):
    value = payload()
    run.check_payload(value, 100000)
    if change == "phase":
        value["format"] = "mobileov_stage1_mcp_v1"
    elif change == "step":
        value["step"] = 15000
    elif change == "nan":
        value["connector"]["weight"][0, 0] = float("nan")
    elif change == "missing_dit":
        value["dit"] = {}
    else:
        del value["contract"]["initial_weights_sha256"]
    with pytest.raises(ValueError):
        run.check_payload(value, 100000)


def test_existing_wrong_checkpoint_not_silently_replaced(tmp_path):
    path = tmp_path / "model.pt"
    value = payload()
    value["step"] = 5
    torch.save(value, path)
    before = run.sha256_file(path)
    with pytest.raises(ValueError, match="expected 100000"):
        run.load_checkpoint(SimpleNamespace(checkpoint=path, expected_step=100000))
    assert run.sha256_file(path) == before


@pytest.mark.parametrize("valid", [True, False])
def test_download_validates_before_atomic_install(tmp_path, monkeypatch, valid):
    import huggingface_hub

    source = tmp_path / "hub.pt"
    value = payload()
    if not valid:
        value["format"] = "wrong"
    torch.save(value, source)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    target = tmp_path / "cache" / "checkpoint.pt"
    args = SimpleNamespace(checkpoint=target, expected_step=100000, hf_repo="owner/repo",
                           hf_file="stage2.pt", hf_revision="pinned-revision")
    if valid:
        result, digest = run.load_checkpoint(args)
        assert result["step"] == 100000 and digest == run.sha256_file(source)
    else:
        with pytest.raises(ValueError, match="Stage-2"):
            run.load_checkpoint(args)
        assert not target.exists()
    assert calls[0]["force_download"] and calls[0]["revision"] == "pinned-revision"
    assert not list(target.parent.glob("*.tmp.*"))


def test_resume_rejects_changed_checkpoint_or_settings(tmp_path):
    items = list(run.sample_items(["A panda"], 1, 2))
    contract = dict(checkpoint_sha256="abc", guidance=7)
    run.bind_run(tmp_path, contract, items)
    run.bind_run(tmp_path, contract, items)
    for change in (dict(checkpoint_sha256="def", guidance=7), dict(checkpoint_sha256="abc", guidance=3)):
        with pytest.raises(ValueError, match="Resume contract differs"):
            run.bind_run(tmp_path, change, items)


def test_untracked_and_extra_videos_rejected(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    extra = videos / "unknown-0.mp4"
    extra.touch()
    items = list(run.sample_items(["A panda"], 1, 0))
    with pytest.raises(ValueError, match="Untracked"):
        run.bind_run(tmp_path, {}, items)
    extra.unlink()
    run.bind_run(tmp_path, {}, items)
    extra.touch()
    with pytest.raises(ValueError, match="Unexpected videos"):
        run.bind_run(tmp_path, {}, items)


def test_duplicate_job_refused(tmp_path):
    with run.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another generator"):
            with run.run_lock(tmp_path):
                pass


def test_cached_video_requires_matching_provenance_hash_and_decode(tmp_path, monkeypatch):
    item = next(run.sample_items(["A panda"], 1, 10))
    video, metadata = tmp_path / item["filename"], tmp_path / "meta.json"
    video.write_bytes(b"video")
    monkeypatch.setattr(run, "decode_check", lambda *args, **kwargs: True)
    run.atomic_json(metadata, dict(sample=item, video_sha256=run.sha256_file(video)))
    assert run.cached_video(video, metadata, item, {})
    assert not run.cached_video(video, metadata, dict(item, seed=11), {})
    monkeypatch.setattr(run, "decode_check", lambda *args, **kwargs: False)
    assert not run.cached_video(video, metadata, item, {})
    video.write_bytes(b"corrupt")
    assert not run.cached_video(video, metadata, item, {})


def test_mp4_decode_checks_every_frame_and_size(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24., (64, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    assert run.decode_check(path, frames=3, width=64, height=64, fps=24)
    assert not run.decode_check(path, frames=49, width=64, height=64, fps=24)
    assert not run.decode_check(path, frames=3, width=128, height=64, fps=24)


def test_load_models_really_installs_both_trained_components(tmp_path, monkeypatch):
    from tools import infer_mobileov_stage1 as infer

    config = tmp_path / "config.yaml"
    config.write_text("test")
    encoder, connector, dit, vae = [torch.nn.Linear(2, 2, bias=False) for _ in range(4)]
    connector.spec = {"width": 2}
    original = copy.deepcopy(dit.state_dict())
    value = payload()
    value["contract"]["config_sha256"] = run.sha256_file(config)
    value["dit"]["weight"] *= 2
    signatures = dict(smolvlm2="s", dit="d", vae="v")
    value["contract"]["initial_weights_sha256"] = signatures
    monkeypatch.setattr(infer, "load_stack", lambda *args: (None, encoder, connector, dit, vae, None))

    def verify(cfg, enc, base_dit, base_vae, contract):
        assert contract["frozen_weights_sha256"] == signatures
        assert torch.equal(base_dit.weight, original["weight"])
        return dict(frozen_weights_verified=True)

    monkeypatch.setattr(infer, "verify_inference_stack", verify)
    *_, audit = run.load_models(SimpleNamespace(config=config), value, torch.device("cpu"))
    assert torch.equal(dit.weight, value["dit"]["weight"])
    assert torch.equal(connector.weight, value["connector"]["weight"])
    assert audit["trained_dit_loaded"] and audit["trained_connector_loaded"]
    assert all(not m.training and not any(p.requires_grad for p in m.parameters())
               for m in (encoder, connector, dit, vae))


def test_cpu_fake_pipeline_generates_and_resumes_without_reloading_models(tmp_path, monkeypatch):
    from tools import infer_mobileov_stage1 as infer
    from diffusers import utils

    info = tmp_path / "info.json"
    run.atomic_json(info, [dict(prompt_en="A panda"), dict(prompt_en="A car")])
    args = run.parse_args(["--vbench-info", str(info), "--output-dir", str(tmp_path / "out")])
    args.output_dir.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-mocked-cuda")
    monkeypatch.setattr(run, "load_checkpoint", lambda _: (payload(), "sha"))
    monkeypatch.setattr(infer, "autocast", lambda _: nullcontext())
    seen, seeds = [], []

    def encoder(prompt, **kwargs):
        seen.append((prompt, kwargs))
        length = 1 if kwargs.get("drop_condition") else 3
        return (torch.ones(1, length, 2), torch.ones(1, length, dtype=torch.bool), torch.ones(1, 2))

    monkeypatch.setattr(run, "load_models", lambda *a: (encoder, lambda *a: a, None, None, None,
                                                       dict(trained_dit_loaded=True)))

    def frames(dit, vae, scheduler, condition, **kwargs):
        assert condition[1].tolist() == [[True, False, False], [True, True, True]]
        assert kwargs["num_frames"] == 49 and kwargs["first_steps"] == 20
        seeds.append(kwargs["seed"])
        return [None] * 49, []

    monkeypatch.setattr(infer, "generate_frames", frames)
    monkeypatch.setattr(utils, "export_to_video", lambda frames, path, fps: Path(path).write_bytes(b"mp4"))
    monkeypatch.setattr(run, "decode_check", lambda *a, **k: True)
    run.generate(args)
    assert len(seen) == 3 and seen[0] == ("", {"drop_condition": True})
    assert seeds == [20260911, 20260912]
    summary = json.loads((args.output_dir / "generation_summary.json").read_text())
    assert summary["videos"] == 2 and not summary["external_anchor"]
    monkeypatch.setattr(run, "load_models", lambda *a: pytest.fail("Completed resume loaded GPU models"))
    run.generate(args)
    assert len(seeds) == 2
    scores = args.output_dir / "scores"
    scores.mkdir()
    (scores / "scene_eval_results.json").write_text("{}")
    (args.output_dir / "videos/A panda-0.mp4").write_bytes(b"broken")
    with pytest.raises(ValueError, match="cached scores exist"):
        run.generate(args)


def test_launcher_uses_stage2_generator_and_full_scoring():
    script = Path(__file__).resolve().parents[1] / "scripts/vbench_mobileov_stage2_100k_1node1gpu.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "#SBATCH --gres=gpu:1" in text
    assert "17519025/stage2_dit_connector_latest.pt" in text
    assert "tools/generate_vbench_mobileov_stage2.py" in text
    assert "tools/evaluate_vbench_resume.py" in text
    assert "tools/report_vbench_low_prompts.py" in text
    assert "generate_vbench_dreamlite" not in text
    assert "install_dreamlite" not in text
    assert '"${MAX_PROMPTS}" -gt 0' in text
