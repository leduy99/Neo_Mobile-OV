from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import av
import pytest
from PIL import Image

from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample, read_media_bytes
from tools.data_prepare import build_stage1_alignment_manifests as build
from tools.data_prepare import download_alignment_images as images
from tools.data_prepare import download_alignment_videos as videos
from tools.data_prepare import prepare_stage1_alignment as pipeline


def encoded_video(frames=60, width=64, height=48, fps=24):
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            image = Image.new("RGB", (width, height), (index * 3 % 255, 40, 80))
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def make_archive(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return images.Shard(path.name, path.stat().st_size, images.sha256_file(path))


POLICY = dict(min_seconds=2, max_seconds=12, min_side=32, min_frames=49)


def test_video_probe_and_full_span_sampling(tmp_path):
    payload = encoded_video()
    info = videos.probe_video(payload, **POLICY)
    assert info["frame_count"] == 60 and info["duration_seconds"] == pytest.approx(2.5)
    assert info["fps"] == 24
    path = tmp_path / "shards/00000/000001.tar"
    shard = make_archive(path, [("./0000001000.mp4", payload)])
    shard = images.Shard("00000/000001.tar", shard.size, shard.sha256)
    receipt = videos.index_shard(path, shard, tmp_path, {"0000001000.mp4": "A changing light."},
                                revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    row = json.loads((tmp_path / receipt["index"]).read_text())
    target = build.media_reference(row, "video", "video")
    record = dict(task="t2v", prompt=row["caption"], target=target)
    sample = read_alignment_sample(record, {"video": tmp_path})
    assert len(sample["target_frames"]) == 49
    assert sample["frame_indices"][0] == 0 and sample["frame_indices"][-1] == 59
    assert sample["frame_times"][-1] > 2.4
    assert sample["conditioning_images"] == []
    assert sample["source_duration_seconds"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="distinct frames"):
        read_alignment_sample(record, {"video": tmp_path}, num_video_frames=61)
    with pytest.raises(ValueError, match="distinct frames"):
        read_alignment_sample(record, {"video": tmp_path}, num_video_frames=1)


@pytest.mark.parametrize("changes,reason", [
    ({"min_seconds": 3}, "duration"), ({"max_seconds": 1}, "too_long"),
    ({"min_frames": 61}, "too_few_frames"), ({"min_side": 128}, "low_resolution"),
])
def test_video_policy_rejects_instead_of_temporal_cropping(changes, reason):
    with pytest.raises(videos.RejectedVideo, match=reason):
        videos.probe_video(encoded_video(), **{**POLICY, **changes})


def test_selection_is_reproducible_and_spans_directories():
    shards = [images.Shard(f"{group:05d}/{index:06d}.tar", 100, "a" * 64)
              for group in range(3) for index in range(10)]
    selected = videos.select_shards(shards, 9, 42)
    assert selected == videos.select_shards(list(reversed(shards)), 9, 42)
    assert len({str(Path(item.path).parent) for item in selected}) == 3
    assert selected != videos.select_shards(shards, 9, 43)


def test_caption_schema_and_duplicate_key_fail(tmp_path):
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps([dict(video="000.mp4", text=" A person dances. ")]))
    assert videos.load_captions(path) == {"000.mp4": "A person dances."}
    path.write_text(json.dumps([dict(video="000.mp4", text="a"), dict(video="000.mp4", text="b")]))
    with pytest.raises(ValueError, match="Duplicate"):
        videos.load_captions(path)
    path.write_text(json.dumps([dict(video="../000.mp4", text="a")]))
    with pytest.raises(ValueError):
        videos.load_captions(path)


def test_index_resume_and_policy_invalidation(tmp_path, monkeypatch):
    archive = tmp_path / "shards/000001.tar"
    shard = make_archive(archive, [("1.mp4", encoded_video())])
    kwargs = dict(revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    receipt = videos.index_shard(archive, shard, tmp_path, {"1.mp4": "A light."}, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(videos, "probe_video", lambda *a, **kw: pytest.fail("resume decoded again"))
        assert videos.index_shard(archive, shard, tmp_path, {"1.mp4": "A light."}, **kwargs) == receipt
    revised = videos.index_shard(archive, shard, tmp_path, {"1.mp4": "A light."},
                                **{**kwargs, "policy": {**POLICY, "min_side": 1000}})
    assert revised["valid_pairs"] == 0 and revised["rejected"] == {"low_resolution": 1}


def test_errors_are_separate_from_intentional_filters(tmp_path):
    archive = tmp_path / "x.tar"
    shard = make_archive(archive, [("bad.mp4", b"broken"), ("missing.mp4", encoded_video()),
                                   ("short.mp4", encoded_video(frames=5))])
    receipt = videos.index_shard(archive, shard, tmp_path, {"bad.mp4": "bad", "short.mp4": "short"},
                                revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    assert receipt["errors"] == {"decode_error": 1, "missing_caption": 1}
    assert receipt["rejected"] == {"too_few_frames": 1}


def test_video_index_disk_failure_is_not_a_bad_sample(tmp_path, monkeypatch):
    archive = tmp_path / "x.tar"
    shard = make_archive(archive, [("1.mp4", encoded_video())])
    original_open = Path.open

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def write(self, _):
            raise OSError("disk quota exceeded")

    def failing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailingWriter(stream) if path.name.endswith(".jsonl.tmp") else stream

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="disk quota"):
        videos.index_shard(archive, shard, tmp_path, {"1.mp4": "A light."},
                           revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    assert not (tmp_path / "indexes/x.tar.json").exists()
    assert not (tmp_path / "indexes/x.tar.jsonl").exists()


@pytest.mark.parametrize("names", [["../1.mp4"], ["./1.mp4", "1.mp4"]])
def test_unsafe_or_duplicate_video_members_fail(tmp_path, names):
    archive = tmp_path / "x.tar"
    shard = make_archive(archive, [(name, encoded_video()) for name in names])
    with pytest.raises(ValueError):
        videos.index_shard(archive, shard, tmp_path, {"1.mp4": "caption"},
                           revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    assert not (tmp_path / "indexes/x.tar.json").exists()


def test_video_download_index_and_offline_resume(tmp_path, monkeypatch):
    source = tmp_path / "fixture/000000.tar"
    shard = make_archive(source, [("1.mp4", encoded_video())])
    annotation = tmp_path / "fixture/annotation.json"
    annotation.write_text(json.dumps([dict(video="1.mp4", text="A changing light.")]))
    spec = images.Shard("annotation.json", annotation.stat().st_size, images.sha256_file(annotation))
    output = tmp_path / "download"
    args = videos.parse_args(["--output-dir", str(output), "--num-shards", "1", "--min-side", "32",
                              "--disk-margin-gib", "0"])
    plan = dict(settings=dict(repo_id=videos.REPO_ID, requested_revision="main", num_shards=1, seed=args.seed),
                revision="pin", annotation=asdict(spec), shards=[asdict(shard)], expected_bytes=spec.size + shard.size)
    monkeypatch.setattr(videos, "resolve_plan", lambda *a, **kw: plan)
    calls = []

    def download(**kwargs):
        calls.append(kwargs["filename"])
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        shutil.copyfile(annotation if kwargs["filename"] == "annotation.json" else source, target)
        return str(target)

    monkeypatch.setattr(images, "hf_hub_download", download)
    summary = videos.run(args)
    assert summary["status"] == "complete" and summary["valid_video_caption_pairs"] == 1
    assert (output / ".download_complete").exists()
    assert videos.run(args) == summary and len(calls) == 2
    args.min_side = 1000
    with pytest.raises(RuntimeError, match="incomplete"):
        videos.run(args)
    assert not (output / ".download_complete").exists()


def source_release(root, rows, count_key):
    root.mkdir(parents=True, exist_ok=True)
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = dict(status="complete", repo_id="fixture", revision="pin", **{count_key: len(rows)},
                   samples_sha256=images.sha256_file(root / "samples.jsonl"))
    images.atomic_json(root / "download_summary.json", summary)
    images.atomic_json(root / ".download_complete", summary)


def fixtures(tmp_path):
    image_root, video_root = tmp_path / "images", tmp_path / "videos"
    payload = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(payload, format="PNG")
    path = image_root / "shards/i.tar"
    shard = make_archive(path, [("1.png", payload.getvalue()), ("1.txt", b"A red square.")])
    receipt = images.index_shard(path, shard, image_root, repo_id="fixture", revision="pin")
    image_row = json.loads((image_root / receipt["index"]).read_text())
    source_release(image_root, [image_row, {**image_row, "sample_id": "duplicate"}], "valid_image_caption_pairs")
    path = video_root / "shards/v.tar"
    shard = make_archive(path, [("1.mp4", encoded_video())])
    receipt = videos.index_shard(path, shard, video_root, {"1.mp4": "A changing light."},
                                revision="pin", annotation_sha256="a" * 64, policy=POLICY)
    video_row = json.loads((video_root / receipt["index"]).read_text())
    source_release(video_root, [video_row], "valid_video_caption_pairs")
    return image_root, video_root


def test_three_tasks_share_images_without_leaking_condition_to_t2i(tmp_path):
    image_root, video_root = fixtures(tmp_path)
    output = tmp_path / "release"
    args = build.parse_args(["--image-root", str(image_root), "--video-root", str(video_root),
                             "--output-dir", str(output), "--validation-fraction", "0"])
    summary = build.run(args)
    assert summary["counts"] == {"train.t2i": 1, "train.t2v": 1, "train.image_reconstruction": 1}
    assert summary["rejected"] == {"blip3o.exact_duplicate": 1}
    assert summary["vae_encoded"] is False and summary["trainer_integrated"] is False
    assert summary["reader_checks"]["train.t2v"] == 1
    assert (output / ".stage1_data_complete").exists()
    sources = load_sources(output)
    tasks = {task: json.loads((output / "train" / f"{task}.jsonl").read_text()) for task in build.TASKS}
    assert tasks["t2i"]["target"] == tasks["image_reconstruction"]["target"]
    t2i = read_alignment_sample(tasks["t2i"], sources)
    reconstruction = read_alignment_sample(tasks["image_reconstruction"], sources)
    t2v = read_alignment_sample(tasks["t2v"], sources)
    assert t2i["conditioning_images"] == [] and t2i["prompt"] == "A red square."
    assert reconstruction["prompt"] == "" and len(reconstruction["conditioning_images"]) == 1
    assert reconstruction["target_frames"][0].tobytes() == reconstruction["conditioning_images"][0].tobytes()
    assert len(t2v["target_frames"]) == 49
    assert build.run(args) == summary


def test_missing_video_source_cannot_claim_stage1_complete(tmp_path):
    image_root, video_root = fixtures(tmp_path)
    (video_root / ".download_complete").unlink()
    output = tmp_path / "release"
    with pytest.raises(RuntimeError, match="not complete"):
        build.run(build.parse_args(["--image-root", str(image_root), "--video-root", str(video_root),
                                    "--output-dir", str(output)]))
    assert not (output / ".stage1_data_complete").exists()


def test_tampered_source_index_is_rejected(tmp_path):
    image_root, video_root = fixtures(tmp_path)
    (image_root / "samples.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="checksum"):
        build.completed_source(image_root, "valid_image_caption_pairs")


def test_split_and_prompt_exclusion(tmp_path):
    checksum = hashlib.sha256(b"same content").hexdigest()
    assert build.split_for("image", checksum, 1, 0.2) == build.split_for("image", checksum, 1, 0.2)
    image_root, video_root = fixtures(tmp_path)
    exclusion = tmp_path / "exclude.txt"
    exclusion.write_text(" A RED    SQUARE.\n")
    output = tmp_path / "release"
    with pytest.raises(RuntimeError, match="No training data for t2i"):
        build.run(build.parse_args(["--image-root", str(image_root), "--video-root", str(video_root),
                                    "--output-dir", str(output), "--validation-fraction", "0",
                                    "--exclude-prompts", str(exclusion)]))
    assert not (output / ".stage1_data_complete").exists()


def test_pipeline_build_only_is_offline_and_covers_three_tasks(tmp_path, monkeypatch):
    image_root, video_root = fixtures(tmp_path)
    monkeypatch.setattr(images, "run", lambda *_: pytest.fail("build-only downloaded images"))
    monkeypatch.setattr(videos, "run", lambda *_: pytest.fail("build-only downloaded videos"))
    output = tmp_path / "release"
    pipeline.main(["--skip-download", "--image-root", str(image_root), "--video-root", str(video_root),
                   "--output-dir", str(output), "--validation-fraction", "0"])
    assert set(json.loads((output / "stage1_summary.json").read_text())["tasks"]) == set(build.TASKS)


def test_stage1_sbatch_syntax_and_archive_reference_safety(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["bash", "-n", str(root / "scripts/prepare_univideo_stage1_data_1node1gpu.sbatch")], check=True)
    with pytest.raises(ValueError, match="Unsafe"):
        read_media_bytes({"a": tmp_path}, dict(source="a", shard="../escape", offset=0, size=1))
