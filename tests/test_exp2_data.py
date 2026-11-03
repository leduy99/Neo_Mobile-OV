import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import pytest

from new_mobile_ov.training import exp2_data as data
from new_mobile_ov.training.stage1_alignment import prepare_sample, verify_release
from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample
from tools.data_prepare import prepare_mobileov_exp2 as pipeline
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file
from tools.data_prepare.run_mobileov_exp2_data import gpu_groups, run_group
from test_stage1_alignment_data import make_archive

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return json.loads((ROOT / "configs/mobileov_exp2_data.json").read_text())


def video_bytes(seed=1, fps=24, frames=96):
    output = io.BytesIO()
    background = np.random.default_rng(seed).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for i in range(frames):
            image = Image.fromarray(np.roll(background, i, axis=1))
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def raw_release(tmp_path):
    parent, media = tmp_path / "parent", tmp_path / "media"
    media.mkdir()
    records, members = [], []
    for i in range(2):
        image = Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 96, 3), dtype=np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        members += [(f"image{i}.png", buf.getvalue()), (f"video{i}.mp4", video_bytes(i + 20))]
    archive = media / "all.tar"
    make_archive(archive, members)
    import tarfile
    with tarfile.open(archive) as source:
        for item, (_, payload) in zip(source.getmembers(), members):
            kind = "image" if item.name.endswith("png") else "video"
            target = dict(source="raw", shard="all.tar", kind=kind, offset=item.offset_data,
                          size=item.size, sha256=hashlib.sha256(payload).hexdigest(), width=96, height=64)
            if kind == "video":
                target.update(fps=24, duration_seconds=4, frame_count=96)
            records.append(dict(task="t2i" if kind == "image" else "t2v", target=target,
                                sample_id=f"{'t2i' if kind == 'image' else 't2v'}:{item.name}",
                                prompt="A cat runs outside and later enters a different room.",
                                conditioning_route="mllm_only"))
    artifacts, counts = {}, {}
    for split, rows in (("train", records[:2]), ("validation", records[2:])):
        (parent / split).mkdir(parents=True)
        for task in ("t2i", "t2v", "image_reconstruction"):
            record = rows[0 if task != "t2v" else 1]
            if task == "image_reconstruction":
                record = dict(record, task=task, sample_id=record["sample_id"].replace("t2i:", f"{task}:"), prompt="")
            path = parent / split / f"{task}.jsonl"
            path.write_text(json.dumps(record) + "\n")
            artifacts[f"{split}.{task}"] = dict(path=str(path.relative_to(parent)), sha256=sha256_file(path))
            counts[f"{split}.{task}"] = 1
    atomic_json(parent / "sources.json", {"raw": dict(root="../media")})
    summary = dict(status="raw_data_ready", tasks=["t2i", "t2v", "image_reconstruction"], counts=counts,
                   manifests=artifacts, sources_sha256=sha256_file(parent / "sources.json"))
    atomic_json(parent / "stage1_summary.json", summary)
    atomic_json(parent / ".stage1_data_complete", summary)
    return parent


class FakeQuality:
    provenance = {"fixture": "quality"}

    def score(self, frames, policy):
        return dict(musiq_mean=70., aesthetic_mean=6., motion_mean_pixels=1. if len(frames) > 1 else None)


class FakeCaption:
    provenance = {"fixture": "caption"}

    def caption(self, frames, policy):
        annotation = data.validate_caption(dict(usable=True, caption_supported=True,
                                                confidence=.95, caption_short="A colorful textured scene",
                                                caption_long="A colorful textured scene fills the visible image.",
                                                motion_observation="object_motion" if len(frames) > 1 else "not_applicable",
                                                text_overlay="none", reason="Only the supplied view is described."), policy)
        return dict(annotation, siglip_logit_mean=5., qwen_model=policy["qwen_model"], qwen_revision="test")


def args_for(parent, output, policy_path, phase="plan", **kwargs):
    args = pipeline.parse_args(["--phase", phase, "--input-root", str(parent), "--output-dir", str(output),
                                "--policy", str(policy_path), "--quality-workers", "2", "--caption-workers", "2",
                                "--reader-checks", "1", "--max-errors", "1"])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("fps", [20, 24, 25, 30, 60])
def test_fixed_fps_preserves_physical_duration(policy, fps):
    times = np.arange(int(5 * fps)) / fps
    view = data.choose_window(times.tolist(), [], policy, "x", "train")
    assert len(view["frame_indices"]) == 49
    assert view["grid_times"][-1] - view["grid_times"][0] == pytest.approx(2)
    assert view["source_frame_times"][-1] - view["source_frame_times"][0] == pytest.approx(2, abs=1 / fps)
    assert view["unique_frames"] >= 40


def test_low_fps_and_timestamp_errors_are_not_sped_up(policy):
    with pytest.raises(data.RejectedSample, match="source_fps"):
        data.choose_window((np.arange(120) / 16).tolist(), [], policy, "x", "train")
    with pytest.raises(data.RejectedSample, match="non_monotonic"):
        data.choose_window([0, .01, .01, 2], [], policy, "x", "train")
    times = (np.arange(120) / 24).tolist()
    times[35:65] = [times[35] + (i + 1) * .00001 for i in range(30)]
    # Every chosen frame must stay on the 24-fps physical grid, or fail explicitly.
    try:
        view = data.choose_window(times, [], policy, "x", "validation")
        assert view["max_timestamp_error_seconds"] <= .04
    except data.RejectedSample:
        pass


def test_windows_never_cross_scene_boundaries(policy):
    times = (np.arange(240) / 24).tolist()
    for index in range(30):
        spec = data.choose_window(times, [72, 144], policy, str(index), "train")
        indices = spec["frame_indices"]
        assert not any(indices[0] < cut <= indices[-1] for cut in [72, 144])
    with pytest.raises(data.RejectedSample, match="single_shot"):
        data.choose_window((np.arange(80) / 24).tolist(), [20, 40, 60], policy, "x", "train")


def test_crop_preserves_aspect_and_is_stable(policy):
    spec = data.spatial_spec(1000, 700, policy, "image", "train")
    rw, rh = spec["resized_size"]
    assert rw / rh == pytest.approx(1000 / 700, abs=.004)
    assert spec == data.spatial_spec(1000, 700, policy, "image", "train")
    image = Image.new("RGB", (1000, 700))
    assert list(data.apply_spatial(image, spec).size) in policy["buckets"]
    assert data.spatial_spec(1000, 700, policy, "image", "validation")["crop_policy"] == "center_crop"
    with pytest.raises(ValueError, match="geometry"):
        data.apply_spatial(Image.new("RGB", (1, 1)), spec)


def test_exif_orientation_is_applied_once_before_caption_and_reconstruction(policy):
    image = Image.new("RGB", (100, 200), "red")
    image.getexif()[274] = 6
    spec = data.spatial_spec(200, 100, policy, "rotated", "validation")
    spec.update(raw_source_size=[100, 200], exif_orientation=6)
    cropped = data.apply_spatial(image, spec)
    assert cropped.size == (512, 320) and cropped.getexif().get(274) is None


@pytest.mark.parametrize("field,value", [("confidence", float("nan")), ("confidence", True),
                                         ("caption_supported", "yes"), ("text_overlay", "unknown"),
                                         ("motion_observation", "fastest_action")])
def test_caption_schema_does_not_silently_accept_bad_outputs(policy, field, value):
    result = dict(usable=True, caption_supported=True, confidence=.9, motion_observation="object_motion",
                  caption_short="A red car", caption_long="A red car moves along the road.", text_overlay="none", reason="visible")
    result[field] = value
    with pytest.raises(ValueError):
        data.validate_caption(result, policy)


def test_caption_changes_required_after_crop_and_temporal_selection(policy):
    prompt = data.caption_instruction(video=True)
    assert "Original caption:" not in prompt and "outside this view" in prompt
    assert "select a different time window" in prompt and "same response" in prompt.lower()
    result = dict(usable=True, caption_supported=True, confidence=.9, motion_observation="static",
                  caption_short="A cat outside", caption_long="A cat stands outside beside a tree.", text_overlay="none", reason="Later scene absent")
    assert data.validate_caption(result, policy, video=True)["accepted"]
    assert not data.validate_caption(dict(result, caption_supported=False), policy)["accepted"]
    assert not data.validate_caption(dict(result, usable=False, caption_short="", caption_long=""), policy)["accepted"]
    assert not data.validate_caption(dict(result, text_overlay="obstructive"), policy)["accepted"]
    assert data.validate_caption(dict(result, text_overlay="minor"), policy)["accepted"]


def test_all_black_is_rejected_but_motion_is_not_aesthetic_truth(policy):
    stats = data.quality_stats([Image.new("RGB", (64, 64))])
    assert "blank_or_saturated" in data.cheap_rejections(stats, policy)
    assert policy["max_motion_pixels"] is None and policy["min_aesthetic"] is None


def test_real_scene_detector_and_reader_pipeline(tmp_path, policy):
    pytest.importorskip("scenedetect")
    parent = raw_release(tmp_path)
    output = tmp_path / "exp2"
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    args = args_for(parent, output, p)
    original_hashes = {str(f): sha256_file(f) for f in parent.rglob("*") if f.is_file()}
    pipeline.build_plan(args)
    for phase, model in (("quality", FakeQuality()), ("caption", FakeCaption())):
        for rank in range(2):
            pipeline.run_worker(args_for(parent, output, p, phase, rank=rank), model=model)
    pipeline.finalize(args)
    summary = verify_release(output)
    assert summary["counts"] == {f"{s}.{t}": 1 for s in ("train", "validation") for t in ("t2i", "t2v", "image_reconstruction")}
    assert summary["semantics_verified"] is False and summary["trainer_integrated"]
    sources = load_sources(output)
    image = json.loads((output / "train/t2i.jsonl").read_text())
    reconstruction = json.loads((output / "train/image_reconstruction.jsonl").read_text())
    video = json.loads((output / "train/t2v.jsonl").read_text())
    assert reconstruction["prompt"] == "" and image["exp2_transform"] == reconstruction["exp2_transform"]
    assert "later enters" not in video["prompt"]
    assert video["original_caption"] == "A cat runs outside and later enters a different room."
    assert video["caption_verification"]["source_caption_provided"] is False
    assert video["caption_verification"]["generation_passes"] == 1
    assert summary["motion_observation_counts"]["t2v.accepted.object_motion"] == 2
    sample = prepare_sample(video, sources)
    assert sample["video"].shape[1] == 49 and sample["effective_sample_fps"] == 24
    assert sample["frame_times"][-1] - sample["frame_times"][0] == pytest.approx(2)
    im = read_alignment_sample(image, sources)["target_frames"][0]
    rec = prepare_sample(reconstruction, sources)
    assert im.tobytes() == rec["image"].tobytes()
    assert rec["video"].shape[1] == 1
    # Idempotent, separate output, no source/archive copying.
    pipeline.finalize(args)
    pipeline.build_plan(args)
    assert original_hashes == {str(f): sha256_file(f) for f in parent.rglob("*") if f.is_file()}
    assert not list(output.rglob("*.mp4")) and not list(output.rglob("*.tar"))
    from tools.data_prepare.render_mobileov_exp2_audit import render
    assert render(output, output / "audit", per_group=1) == 2
    page = (output / "audit/index.html").read_text()
    assert "Original:" in page and "Replacement:" in page
    assert len(list((output / "audit").glob("*.jpg"))) == 2
    audit_video = next((output / "audit").glob("*.mp4"))
    with av.open(str(audit_video)) as container:
        assert container.streams.video[0].average_rate == 24
        assert sum(1 for _ in container.decode(video=0)) == 49
    for phase in ("quality", "caption"):
        pipeline.run_worker(args_for(parent, output, p, phase, rank=0), model=None)
    changed = dict(video)
    changed["exp2_transform"] = dict(video["exp2_transform"], format="tampered")
    with pytest.raises(ValueError, match="modified"):
        read_alignment_sample(changed, sources)
    policy["short_caption_fraction"] = .5
    p.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="NEW directory"):
        pipeline.build_plan(args)


def test_runtime_failure_cannot_be_frozen_as_rejection(tmp_path, policy, monkeypatch):
    parent = raw_release(tmp_path)
    output, p = tmp_path / "exp2", tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    args = args_for(parent, output, p)
    pipeline.build_plan(args)
    monkeypatch.setattr(pipeline, "make_view", lambda *a: (_ for _ in ()).throw(RuntimeError("decoder failure")))
    with pytest.raises(RuntimeError, match="runtime/model"):
        pipeline.run_worker(args_for(parent, output, p, "quality"), model=FakeQuality())
    assert not (output / "work/quality/worker_000.complete.json").exists()
    with pytest.raises(RuntimeError, match="Incomplete"):
        pipeline.finalize(args)
    monkeypatch.undo()
    pipeline.run_worker(args_for(parent, output, p, "quality"), model=FakeQuality())
    path = output / "work/quality/worker_000.sqlite"
    import sqlite3
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM results WHERE status='accepted'").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM results WHERE status='error'").fetchone()[0] == 0


def test_output_cannot_replace_old_data(tmp_path, policy):
    parent = raw_release(tmp_path)
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    for output in (parent, parent / "exp2", tmp_path / "media" / "exp2"):
        with pytest.raises(ValueError, match="separate|source media"):
            pipeline.build_plan(args_for(parent, output, p))


def test_gpu_sharding_logs_and_job_contract():
    assert gpu_groups(["0", "1", "2", "3"], 2) == [["0", "1"], ["2", "3"]]
    with pytest.raises(ValueError):
        gpu_groups(["0", "1", "2"], 2)
    script = (ROOT / "scripts/prepare_mobileov_exp2_data_1node8gpu.sbatch").read_text()
    assert "--gres=gpu:8" in script and "--target" in script
    assert "neo_image_data" in script and "neo_vbench" in script
    assert "mobileov_exp2_visual_only_v2" in script and "afterany:" in script
    subprocess.run(["bash", "-n", str(ROOT / "scripts/prepare_mobileov_exp2_data_1node8gpu.sbatch")], check=True)
    smoke = ROOT / "scripts/smoke_mobileov_exp2_data_1node8gpu.sbatch"
    subprocess.run(["bash", "-n", str(smoke)], check=True)
    assert "mobileov_exp2_preview" in smoke.read_text() and "AUTO_CONTINUE=0" in smoke.read_text()
    local = ROOT / "scripts/smoke_mobileov_exp2_visual_only_local_h200.sbatch"
    subprocess.run(["bash", "-n", str(local)], check=True)
    assert "--gres=gpu:1" in local.read_text() and "--caption-gpus-per-worker 1" in local.read_text()


def test_launcher_stops_peers_on_error_and_deadline(tmp_path):
    import os
    import time
    env = dict(os.environ)
    with pytest.raises(RuntimeError, match="worker failed"):
        run_group([([sys.executable, "-c", "raise SystemExit(1)"], env, tmp_path / "error.log"),
                   ([sys.executable, "-c", "import time; time.sleep(30)"], env, tmp_path / "other.log")])
    assert run_group([([sys.executable, "-c", "import time; time.sleep(30)"], env, tmp_path / "pause.log")],
                     deadline=time.monotonic() + .1) == 75


def test_exact_24fps_center_window_never_duplicates_due_to_ties(policy):
    view = data.choose_window((np.arange(96) / 24).tolist(), [], policy, "x", "validation")
    assert view["unique_frames"] == 49
    assert set(np.diff(view["frame_indices"])) == {1}


def test_changed_source_paths_are_rejected(tmp_path, policy):
    parent = raw_release(tmp_path)
    p, output = tmp_path / "policy.json", tmp_path / "exp2"
    p.write_text(json.dumps(policy))
    args = args_for(parent, output, p)
    pipeline.build_plan(args)
    atomic_json(output / "sources.json", {"raw": {"root": "../elsewhere"}})
    with pytest.raises(ValueError, match="source roots"):
        pipeline.load_contract(args)


def test_preview_cannot_be_mistaken_for_a_training_release(tmp_path):
    parent = raw_release(tmp_path)
    summary = json.loads((parent / "stage1_summary.json").read_text())
    summary.update(format=data.FORMAT, preview_only=True)
    atomic_json(parent / "stage1_summary.json", summary)
    atomic_json(parent / ".stage1_data_complete", summary)
    with pytest.raises(ValueError, match="preview"):
        verify_release(parent)


def test_model_changes_are_detected(tmp_path):
    pipeline.verify_models(tmp_path, "quality", {"raft": "old"})
    with pytest.raises(ValueError, match="weights/revisions changed"):
        pipeline.verify_models(tmp_path, "quality", {"raft": "new"})


def test_model_loader_reuses_local_snapshots_instead_of_network_ids():
    code = (ROOT / "tools/data_prepare/exp2_models.py").read_text()
    assert "local_files_only=True" in code
    assert "model_id=str(siglip_path)" in code
    assert "model_id=str(qwen_path)" in code
    assert "paddleocr" not in code


def test_real_scene_cut_is_excluded_from_selected_window(policy):
    pytest.importorskip("scenedetect")
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for i in range(120):
            image = Image.new("RGB", (96, 64), "red" if i < 60 else "blue")
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    times, cuts, size = data.video_probe(output.getvalue(), policy)
    assert 60 in cuts and size == [96, 64]
    spec = data.choose_window(times, cuts, policy, "cut", "train")
    assert not (spec["frame_indices"][0] < 60 <= spec["frame_indices"][-1])


def test_variable_frame_timestamps_remain_on_physical_grid(policy):
    base = np.arange(150) / 24
    times = base + .004 * np.sin(np.arange(150))
    view = data.choose_window(times.tolist(), [], policy, "variable", "validation")
    assert view["unique_frames"] == 49
    assert max(abs(a - b) for a, b in zip(view["source_frame_times"], view["grid_times"])) < .04


def test_caption_model_sees_the_crop_sequence_not_just_frame_zero(policy):
    import torch
    from types import SimpleNamespace
    from tools.data_prepare.exp2_models import CaptionModels

    payload = dict(usable=True, caption_supported=True, confidence=.9, motion_observation="object_motion",
                   caption_short="A red car", caption_long="A red car moves along the road.", text_overlay="none", reason="visible")
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return {"input_ids": torch.tensor([[1, 2]])}
        def batch_decode(self, ids, **kwargs):
            return [json.dumps(payload)]
    class Model:
        calls = 0
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))
        def generate(self, **kwargs):
            self.calls += 1
            return torch.tensor([[1, 2, 3]])
    processor = Processor()
    instance = CaptionModels.__new__(CaptionModels)
    instance.verifier = SimpleNamespace(processor=processor, model=Model())
    instance.siglip = SimpleNamespace(score=lambda images, texts: ([.9] * len(images), [3.] * len(images)))
    instance.provenance = dict(qwen_model=policy["qwen_model"], qwen_revision="test")
    frames = [Image.new("RGB", (512, 320), (i, 0, 0)) for i in range(49)]
    result = instance.caption(frames, policy)
    content = processor.messages[0]["content"]
    assert len([part for part in content if part["type"] == "image"]) == 13
    assert [part for part in content if part["type"] == "image"][-1]["image"] is frames[-1]
    assert any(part.get("text") == "Time 2.000 seconds:" for part in content)
    assert result["accepted"] and result["siglip_logit_mean"] == 3
    assert instance.verifier.model.calls == 1
    assert all("Original caption:" not in part.get("text", "") for part in content)
    payload.update(usable=False, caption_short="", caption_long="", confidence=.2)
    instance.siglip.score = lambda *a: (_ for _ in ()).throw(AssertionError("Rejected samples need no SigLIP pass"))
    rejected = instance.caption(frames, policy)
    assert not rejected["accepted"] and rejected["siglip_logit_mean"] is None
    assert instance.verifier.model.calls == 2  # Exactly one generation for each attempted sample.


def test_small_single_gpu_cannot_load_both_caption_models(policy, monkeypatch):
    import torch
    from types import SimpleNamespace
    from tools.data_prepare.exp2_models import CaptionModels
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=80 * 1024**3))
    with pytest.raises(RuntimeError, match="two GPUs"):
        CaptionModels(policy)
