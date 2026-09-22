from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from test_stage1_alignment_data import POLICY, encoded_video, make_archive
from tools.data_prepare import build_stage1_alignment_manifests as build
from tools.data_prepare import download_alignment_images as images
from tools.data_prepare import download_alignment_videos as videos
from tools.data_prepare import expand_stage1_video_data as expand
from new_mobile_ov.training.stage1_alignment import verify_release
from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample

PIN = "a" * 40


def finish_source(root, rows, **fields):
    root.mkdir(parents=True, exist_ok=True)
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = dict(status="complete", samples_sha256=images.sha256_file(root / "samples.jsonl"), **fields)
    images.atomic_json(root / "download_summary.json", report)
    images.atomic_json(root / ".download_complete", report)
    return report


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    origin, old, image_root, base = (tmp_path / x for x in ("origin", "old", "images", "base"))
    captions, shards, paths = {}, [], {}
    for i in range(6):
        name = f"{i % 2:05d}/{i:06d}.tar"
        members = []
        for j in range(4):
            key = f"{i:03d}_{j:03d}.mp4"
            captions[key] = f"A moving light {i} {j}."
            members.append((key, encoded_video(frames=50 + i * 4 + j)))
        path = origin / name
        spec = make_archive(path, members)
        shards.append(images.Shard(name, spec.size, spec.sha256))
        paths[name] = path
    annotation = origin / "annotation.json"
    annotation.write_text(json.dumps([dict(video=k, text=v) for k, v in captions.items()]))
    ann_spec = images.Shard("annotation.json", annotation.stat().st_size, images.sha256_file(annotation))
    paths["annotation.json"] = annotation
    old_plan = dict(settings=dict(repo_id=videos.REPO_ID, requested_revision="main", num_shards=2, seed=20260911),
                    revision=PIN, annotation=asdict(ann_spec), shards=[asdict(s) for s in shards[:2]],
                    expected_bytes=ann_spec.size + sum(s.size for s in shards[:2]))
    old_rows = []
    for shard in shards[:2]:
        destination = old / "shards" / shard.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[shard.path], destination)
        receipt = videos.index_shard(destination, shard, old, captions, revision=PIN,
                                     annotation_sha256=ann_spec.sha256, policy=POLICY)
        old_rows.extend(json.loads(line) for line in (old / receipt["index"]).read_text().splitlines())
    shutil.copyfile(annotation, old / "shards/annotation.json")
    images.atomic_json(old / "download_plan.json", old_plan)
    finish_source(old, old_rows, repo_id=videos.REPO_ID, revision=PIN,
                  valid_video_caption_pairs=len(old_rows), annotation_sha256=ann_spec.sha256,
                  policy=POLICY, archive_bytes=old_plan["expected_bytes"])
    members = []
    for i in range(10):
        payload = io.BytesIO()
        Image.new("RGB", (64, 48), (i * 20, 40, 80)).save(payload, format="PNG")
        members.extend([(f"{i}.png", payload.getvalue()), (f"{i}.txt", f"Square {i}.".encode())])
    image_path = image_root / "shards/images.tar"
    image_spec = make_archive(image_path, members)
    receipt = images.index_shard(image_path, image_spec, image_root, repo_id="images", revision=PIN)
    image_rows = [json.loads(line) for line in (image_root / receipt["index"]).read_text().splitlines()]
    finish_source(image_root, image_rows, repo_id="images", revision=PIN, valid_image_caption_pairs=len(image_rows))
    build.run(build.parse_args(["--image-root", str(image_root), "--video-root", str(old),
                               "--output-dir", str(base), "--validation-fraction", "0.3"]))
    verify_release(base)
    downloads = []

    class API:
        def dataset_info(self, repo_id, revision):
            assert repo_id == videos.REPO_ID and revision == PIN
            return SimpleNamespace(sha=PIN)

        def list_repo_tree(self, *a, **kw):
            return [SimpleNamespace(path=s.path, size=s.size, lfs=SimpleNamespace(sha256=s.sha256))
                    for s in [ann_spec, *shards]]

    def download(**kwargs):
        assert kwargs["revision"] == PIN
        name = kwargs["filename"]
        downloads.append(name)
        target = Path(kwargs["local_dir"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], target)
        return str(target)

    monkeypatch.setattr(videos, "HfApi", API)
    monkeypatch.setattr(images, "hf_hub_download", download)
    args = expand.parse_args(["--base-root", str(base), "--video-root", str(old),
                              "--extra-video-root", str(tmp_path / "extra"),
                              "--output-dir", str(tmp_path / "expanded"), "--total-video-shards", "5",
                              "--min-train-videos", "5", "--reader-check-samples", "1",
                              "--disk-margin-gib", "0.001", "--workers", "2"])
    return SimpleNamespace(args=args, base=base, old=old, shards=shards, downloads=downloads,
                           old_rows=old_rows, annotation=ann_spec)


def snapshot(root):
    return {str(p.relative_to(root)): (images.sha256_file(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_end_to_end_expansion_reuses_media_and_freezes_validation(corpus):
    c = corpus
    old_before, base_before = snapshot(c.old), snapshot(c.base)
    summary = expand.run(c.args)
    verify_release(c.args.output_dir)
    assert summary["counts"]["train.t2v"] >= 5
    assert summary["expansion"]["added_train_videos"] > 0
    assert summary["expansion"]["added_reader_checks"] == 1
    assert snapshot(c.old) == old_before and snapshot(c.base) == base_before
    original_paths = {s.path for s in c.shards[:2]}
    assert len(c.downloads) == 3 and not original_paths.intersection(c.downloads)
    assert "annotation.json" not in c.downloads
    for split in ("train", "validation"):
        for task in build.TASKS:
            if split == "validation" or task != "t2v":
                assert (c.base / split / f"{task}.jsonl").read_bytes() == (
                    c.args.output_dir / split / f"{task}.jsonl").read_bytes()
    assert (c.args.output_dir / "train/t2v.jsonl").read_bytes().startswith((c.base / "train/t2v.jsonl").read_bytes())
    sources = load_sources(c.args.output_dir)
    assert sources["vchitect"] == c.old
    assert sources["vchitect_added"] == c.args.extra_video_root
    train = [json.loads(line) for line in (c.args.output_dir / "train/t2v.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (c.args.output_dir / "validation/t2v.jsonl").read_text().splitlines()]
    heldout = [json.loads(line) for line in (c.args.output_dir / "heldout/t2v_added.jsonl").read_text().splitlines()]
    assert {r["target"]["sha256"] for r in train}.isdisjoint({r["target"]["sha256"] for r in [*val, *heldout]})
    assert len({r["sample_id"] for r in train}) == len(train)
    for row in train:
        assert len(read_alignment_sample(row, sources)["target_frames"]) == 49


def test_offline_resume_does_not_redownload_or_redecode(corpus, monkeypatch):
    expected = expand.run(corpus.args)
    calls = len(corpus.downloads)
    monkeypatch.setattr(videos, "HfApi", lambda: pytest.fail("resume queried catalog"))
    monkeypatch.setattr(videos, "probe_video", lambda *a, **kw: pytest.fail("resume decoded archive again"))
    assert expand.run(corpus.args) == expected
    assert len(corpus.downloads) == calls


def test_failed_shard_resume_keeps_completed_indexes(corpus, monkeypatch):
    original_index = videos.index_shard
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("Simulated interruption")
        return original_index(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(videos, "index_shard", fail_once)
        with pytest.raises(RuntimeError, match="incomplete"):
            expand.run(corpus.args)
    assert not (corpus.args.output_dir / ".stage1_data_complete").exists()
    assert not (corpus.args.extra_video_root / ".download_complete").exists()
    calls = len(corpus.downloads)
    good_indexes = snapshot(corpus.args.extra_video_root / "indexes")
    result = expand.run(corpus.args)
    assert result["expansion"]["status"] == "complete" and len(corpus.downloads) == calls
    after = snapshot(corpus.args.extra_video_root / "indexes")
    assert all(after[key] == value for key, value in good_indexes.items())


def test_metadata_only_dry_run_downloads_nothing(corpus):
    corpus.args.dry_run = True
    result = expand.run(corpus.args)
    assert result["old_shards"] == 2 and result["additional_shards"] == 3
    assert corpus.downloads == []
    assert not corpus.args.extra_video_root.exists()
    assert not (corpus.args.output_dir / ".stage1_data_complete").exists()


def test_total_budget_counts_old_and_added_archives(corpus):
    corpus.args.max_total_video_gib = 0.000001
    with pytest.raises(RuntimeError, match="budget exceeded"):
        expand.run(corpus.args)
    assert corpus.downloads == []


def test_too_few_videos_never_produces_ready_release(corpus):
    corpus.args.min_train_videos = 100000
    with pytest.raises(RuntimeError, match="training videos; need"):
        expand.run(corpus.args)
    assert (corpus.args.extra_video_root / ".download_complete").exists()
    assert not (corpus.args.output_dir / ".stage1_data_complete").exists()
    assert json.loads((corpus.args.output_dir / "expansion_report.json").read_text())["status"] == "insufficient_videos"


def test_original_source_damage_fails_before_download(corpus):
    path = corpus.old / "shards" / corpus.shards[0].path
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="missing or wrong size"):
        expand.run(corpus.args)
    assert corpus.downloads == []


def test_plan_changes_do_not_overwrite_existing_selection(corpus):
    expand.run(corpus.args)
    path = corpus.args.extra_video_root / "download_plan.json"
    before = path.read_bytes()
    corpus.args.total_video_shards = 6
    with pytest.raises(ValueError, match="plan differs"):
        expand.run(corpus.args)
    assert path.read_bytes() == before


def test_tampered_base_manifests_fail_before_download(corpus):
    with (corpus.base / "validation/t2v.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="Base manifest changed"):
        expand.run(corpus.args)
    assert corpus.downloads == []


def test_overlapping_output_rejected(corpus):
    corpus.args.output_dir = corpus.base
    with pytest.raises(ValueError, match="separate"):
        expand.run(corpus.args)
    corpus.args.output_dir = corpus.base.parent / "images" / "expanded"
    with pytest.raises(ValueError, match="overlap"):
        expand.run(corpus.args)


def test_pinned_revision_and_disjoint_plan(corpus, tmp_path):
    plan = videos.resolve_plan(tmp_path / "unused", revision=PIN, count=3, seed=42,
                               exclude_plan=corpus.old / "download_plan.json")
    assert len(plan["shards"]) == 3
    assert not {s["path"] for s in plan["shards"]}.intersection(s.path for s in corpus.shards[:2])
    assert plan["settings"]["excluded_plan_sha256"] == images.sha256_file(corpus.old / "download_plan.json")
    with pytest.raises(ValueError, match="pinned revision"):
        videos.resolve_plan(tmp_path / "unused", revision="main", count=3, seed=42,
                            exclude_plan=corpus.old / "download_plan.json")


def test_corrupted_cached_plan_cannot_download_original_shards(corpus):
    expand.run(corpus.args)
    path = corpus.args.extra_video_root / "download_plan.json"
    plan = json.loads(path.read_text())
    plan["shards"][0] = asdict(corpus.shards[0])
    images.atomic_json(path, plan)
    with pytest.raises(ValueError, match="overlaps or differs"):
        expand.run(corpus.args)


def test_deduplication_includes_old_validation_and_new_rows(corpus):
    expand.run(corpus.args)
    extra = corpus.args.extra_video_root
    report = json.loads((extra / "download_summary.json").read_text())
    rows = [json.loads(line) for line in (extra / "samples.jsonl").read_text().splitlines()]
    train = json.loads((corpus.base / "train/t2v.jsonl").read_text().splitlines()[0])
    val = json.loads((corpus.base / "validation/t2v.jsonl").read_text().splitlines()[0])
    # References need not resolve because rejected rows must never reach media decoding.
    rows.extend([
        {**rows[0], "sample_id": "copy_train", "video_sha256": train["target"]["sha256"]},
        {**rows[0], "sample_id": "copy_val", "video_sha256": val["target"]["sha256"]},
        {**rows[0], "sample_id": "same_caption", "caption": "  " + val["prompt"].upper() + "  "},
        rows[0],
    ])
    finish_source(extra, rows, **{k: v for k, v in report.items() if k not in
                                 ("status", "samples_sha256", "valid_video_caption_pairs")},
                  valid_video_caption_pairs=len(rows))
    new_output = corpus.base.parent / "dedup_output"
    summary = expand.build_expanded_release(corpus.base, extra, new_output, min_train_videos=5, reader_checks=1)
    assert summary["expansion"]["rejected"]["duplicate_media_or_id"] == 3
    assert summary["expansion"]["rejected"]["matches_old_validation_prompt"] == 1
    verify_release(new_output)


def test_original_scripts_remain_valid_and_new_job_has_safe_defaults():
    root = Path(__file__).resolve().parents[1]
    for name in ("expand_univideo_stage1_videos_1node1gpu.sbatch", "run_alignment_image_download.sh",
                 "prepare_univideo_stage1_data_1node1gpu.sbatch"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    args = expand.parse_args([])
    assert args.total_video_shards == 650 and args.min_train_videos == 500000
    assert args.extra_video_root != args.video_root and args.output_dir != args.base_root
    script = (root / "scripts/expand_univideo_stage1_videos_1node1gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:1" in script and "stage1-expand" in script


@pytest.mark.parametrize("scope,tool,flag,value", [
    ("stage1-expand", "expand_stage1_video_data.py", "--total-video-shards", "650"),
    ("stage1", "prepare_stage1_alignment.py", "--video-shards", "100"),
    ("images", "download_alignment_images.py", "--num-shards", "100"),
])
def test_shell_routes_and_no_gpu_during_dry_run(tmp_path, scope, tool, flag, value):
    root = Path(__file__).resolve().parents[1]
    stub = tmp_path / "python_stub"
    stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint('STUB_ARGS=' + json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    env = {**os.environ, "ALIGNMENT_SCOPE": scope, "DRY_RUN": "1", "PYTHON_BIN": str(stub),
           "DATASET_OUTPUT_DIR": str(tmp_path / "out"), "HF_HOME": str(tmp_path / "hf"),
           "TMPDIR": str(tmp_path / "tmp")}
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(root / "scripts/run_alignment_image_download.sh")],
                            env=env, capture_output=True, text=True, check=True)
    commands = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith("STUB_ARGS=")]
    command = commands[-1]
    assert command[0] == f"tools/data_prepare/{tool}"
    assert command[command.index(flag) + 1] == value and "--dry-run" in command
    assert not any("gpu_heartbeat.py" in str(c) for c in commands)
    if scope == "stage1-expand":
        assert command[command.index("--min-train-videos") + 1] == "500000"
        assert command[command.index("--workers") + 1] == "8"


def test_sbatch_wrapper_forwards_expansion_overrides(tmp_path):
    root = Path(__file__).resolve().parents[1]
    stub = tmp_path / "python_stub"
    stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint('STUB_ARGS=' + json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    module = tmp_path / "module"
    module.write_text("#!/bin/bash\nexit 0\n")
    module.chmod(0o755)
    env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": str(stub), "SLURM_SUBMIT_DIR": str(root),
           "PATH": f"{tmp_path}:{os.environ['PATH']}", "HF_HOME": str(tmp_path / "hf"),
           "TMPDIR": str(tmp_path / "tmp"), "DATASET_OUTPUT_DIR": str(tmp_path / "output"),
           "TOTAL_VIDEO_SHARDS": "800", "MIN_TRAIN_VIDEOS": "600000", "DOWNLOAD_WORKERS": "12"}
    result = subprocess.run(["bash", str(root / "scripts/expand_univideo_stage1_videos_1node1gpu.sbatch")],
                            env=env, capture_output=True, text=True, check=True)
    command = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
               if line.startswith("STUB_ARGS=")][-1]
    assert command[0].endswith("expand_stage1_video_data.py")
    assert command[command.index("--total-video-shards") + 1] == "800"
    assert command[command.index("--min-train-videos") + 1] == "600000"
    assert command[command.index("--workers") + 1] == "12"
