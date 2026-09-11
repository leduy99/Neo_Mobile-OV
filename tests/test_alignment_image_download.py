from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest
from PIL import Image

from tools.data_prepare import download_alignment_images as data


def image_bytes(mode="RGB"):
    target = io.BytesIO()
    Image.new(mode, (12, 8)).save(target, format="PNG")
    return target.getvalue()


def make_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return data.Shard(path.name, path.stat().st_size, data.sha256_file(path))


def valid_archive(tmp_path, name="sa_000001.tar"):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    path = source / name
    shard = make_tar(path, [("nested/a.txt", b"A red cup.\n"), ("nested/a.png", image_bytes()),
                            ("b.png", image_bytes("L")), ("b.txt", b"A gray cup.")])
    return path, shard


def test_selection_is_proportional_order_independent_and_seeded():
    shards = [data.Shard(f"{group}_{i:06d}.tar", 100, "a" * 64)
              for group, count in [("sa", 100), ("webdataset_JDB", 40), ("webdataset_shard", 150)]
              for i in range(count)]
    selected = data.select_shards(shards, 29, 12)
    assert data.Counter(data.shard_group(item.path) for item in selected) == {
        "sa": 10, "webdataset_JDB": 4, "webdataset_shard": 15}
    assert selected == data.select_shards(list(reversed(shards)), 29, 12)
    assert selected != data.select_shards(shards, 29, 13)
    assert len({item.path for item in selected}) == 29


@pytest.mark.parametrize("count", [0, 2])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        data.select_shards([data.Shard("sa_1.tar", 1, "a" * 64)], count, 1)


@pytest.mark.parametrize("path,checksum", [("../evil.tar", "a" * 64), ("sa_1.tar", "")])
def test_invalid_remote_metadata(path, checksum):
    with pytest.raises(ValueError):
        data.select_shards([data.Shard(path, 1, checksum)], 1, 1)


def test_pinned_plan_resumes_without_contacting_main(tmp_path, monkeypatch):
    settings = dict(repo_id=data.REPO_ID, requested_revision="main", num_shards=1, seed=1)
    expected = dict(settings=settings, revision="original-commit")
    data.atomic_json(tmp_path / "download_plan.json", expected)
    monkeypatch.setattr(data, "HfApi", lambda: pytest.fail("must reuse the pinned plan"))
    assert data.resolve_plan(tmp_path, repo_id=data.REPO_ID, revision="main", count=1, seed=1) == expected
    with pytest.raises(ValueError, match="new output directory"):
        data.resolve_plan(tmp_path, repo_id=data.REPO_ID, revision="main", count=1, seed=2)


def test_correct_size_corrupt_archive_forces_redownload(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    output = tmp_path / "output"
    (output / "shards").mkdir(parents=True)
    target = output / "shards" / shard.path
    target.write_bytes(b"x" * shard.size)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        shutil.copyfile(source, target)
        return str(target)

    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    assert data.download_shard(shard, output=output, repo_id=data.REPO_ID, revision="pin", retries=2) == target
    assert len(calls) == 1 and calls[0]["force_download"]
    assert calls[0]["repo_type"] == "dataset" and calls[0]["revision"] == "pin"
    data.download_shard(shard, output=output, repo_id=data.REPO_ID, revision="pin", retries=2)
    assert len(calls) == 1


def test_retry_rejects_wrong_hash_even_when_download_reports_success(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    output = tmp_path / "output"
    (output / "shards").mkdir(parents=True)
    target = output / "shards" / shard.path
    calls = []

    def download(**kwargs):
        calls.append(kwargs["force_download"])
        if len(calls) == 1:
            target.write_bytes(b"x" * shard.size)
        else:
            shutil.copyfile(source, target)
        return str(target)

    monkeypatch.setattr(data, "hf_hub_download", download)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    data.download_shard(shard, output=output, repo_id=data.REPO_ID, revision="pin", retries=2)
    assert calls == [False, True]


def test_tar_index_roundtrip_without_extraction_and_resume(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    (tmp_path / "shards").mkdir()
    archive = tmp_path / "shards" / shard.path
    shutil.copyfile(source, archive)
    receipt = data.index_shard(archive, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert receipt["valid_pairs"] == 2 and receipt["rejected"] == {}
    records = [json.loads(row) for row in (tmp_path / receipt["index"]).read_text().splitlines()]
    assert records[0]["caption"] == "A red cup."
    for record in records:
        assert data.load_indexed_image(tmp_path, record).size == (12, 8)
    assert not (tmp_path / "nested").exists()
    with monkeypatch.context() as scoped:
        scoped.setattr(data.Image, "open", lambda *_: pytest.fail("valid cached index must not re-decode"))
        assert data.index_shard(archive, shard, tmp_path, repo_id=data.REPO_ID, revision="pin") == receipt
    (tmp_path / receipt["index"]).write_text("broken index")
    repaired = data.index_shard(archive, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert repaired["index_sha256"] == receipt["index_sha256"]


def test_hub_partial_failure_forces_clean_retry(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    output = tmp_path / "output"
    (output / "shards").mkdir(parents=True)
    target = output / "shards" / shard.path
    calls = []

    def download(**kwargs):
        calls.append(kwargs["force_download"])
        if not kwargs["force_download"]:
            raise OSError("Consistency check failed: oversized partial; HTTP 416")
        shutil.copyfile(source, target)
        return str(target)

    monkeypatch.setattr(data, "hf_hub_download", download)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    data.download_shard(shard, output=output, repo_id=data.REPO_ID, revision="pin", retries=2)
    assert calls == [False, True]


def test_index_write_failure_is_not_a_sample_rejection(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    original_open = Path.open

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def write(self, value):
            self.stream.write(value[:10])
            raise OSError("disk quota exceeded")

    def failing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailingWriter(stream) if path.name.endswith(".jsonl.tmp") else stream

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="disk quota"):
            data.index_shard(source, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert not (tmp_path / "indexes" / f"{shard.path}.json").exists()
    assert not (tmp_path / "indexes" / f"{shard.path}.jsonl").exists()
    receipt = data.index_shard(source, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert receipt["valid_pairs"] == 2 and receipt["rejected"] == {}


def test_bad_samples_are_counted_not_silently_accepted(tmp_path):
    archive = tmp_path / "sa_1.tar"
    shard = make_tar(archive, [("ok.png", image_bytes()), ("ok.txt", b"A cup."),
                               ("empty.png", image_bytes()), ("empty.txt", b" \n"),
                               ("bad.jpg", b"not an image"), ("bad.txt", b"A dog."),
                               ("orphan.png", image_bytes())])
    receipt = data.index_shard(archive, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert receipt["valid_pairs"] == 1
    assert receipt["rejected"] == {"unpaired_member": 1, "invalid_image_or_caption": 2}


@pytest.mark.parametrize("members", [
    [("../a.png", image_bytes()), ("../a.txt", b"unsafe")],
    [("a.png", image_bytes()), ("a.png", image_bytes()), ("a.txt", b"duplicate")],
    [("a.txt", b"no image")],
])
def test_bad_archive_cannot_get_success_receipt(tmp_path, members):
    path = tmp_path / "sa_1.tar"
    shard = make_tar(path, members)
    with pytest.raises((ValueError, RuntimeError)):
        data.index_shard(path, shard, tmp_path, repo_id=data.REPO_ID, revision="pin")
    assert not (tmp_path / "indexes/sa_1.tar.json").exists()


def test_same_output_is_exclusive(tmp_path):
    with data.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another downloader"):
            with data.output_lock(tmp_path):
                pass


def fixture_plan(shard, args):
    return dict(settings=dict(repo_id=args.repo_id, requested_revision=args.revision,
                              num_shards=args.num_shards, seed=args.seed),
                revision="pin", expected_bytes=shard.size, shards=[asdict(shard)])


def test_full_download_index_resume_and_failure_marker(tmp_path, monkeypatch):
    source, shard = valid_archive(tmp_path)
    output = tmp_path / "output"
    args = data.parse_args(["--output-dir", str(output), "--num-shards", "1", "--disk-margin-gib", "0"])
    plan = fixture_plan(shard, args)
    monkeypatch.setattr(data, "resolve_plan", lambda *a, **kw: plan)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        shutil.copyfile(source, target)
        return str(target)

    monkeypatch.setattr(data, "hf_hub_download", download)
    summary = data.run(args)
    assert summary["status"] == "complete" and summary["valid_image_caption_pairs"] == 2
    assert (output / ".download_complete").is_file()
    assert len((output / "samples.jsonl").read_text().splitlines()) == 2
    assert data.run(args) == summary
    assert len(calls) == 1
    monkeypatch.setattr(data, "download_shard", lambda *a, **kw: (_ for _ in ()).throw(OSError("network")))
    with pytest.raises(RuntimeError, match="incomplete"):
        data.run(args)
    assert not (output / ".download_complete").exists()
    assert json.loads((output / "download_summary.json").read_text())["status"] == "incomplete"


def test_dry_run_and_budget_do_not_download(tmp_path, monkeypatch):
    _, shard = valid_archive(tmp_path)
    output = tmp_path / "output"
    args = data.parse_args(["--output-dir", str(output), "--num-shards", "1", "--dry-run"])
    plan = fixture_plan(shard, args)
    monkeypatch.setattr(data, "resolve_plan", lambda *a, **kw: plan)
    monkeypatch.setattr(data, "hf_hub_download", lambda **kw: pytest.fail("dry run downloaded media"))
    data.run(args)
    assert not (output / ".download_complete").exists()
    assert not (output / "download_plan.json").exists()
    args.max_download_gib = 1e-9
    with pytest.raises(RuntimeError, match="max-download-gib"):
        data.run(args)


def test_shell_syntax_and_no_gpu_outside_slurm(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_alignment_image_download.sh"
    for shell_file in (script, root / "scripts/download_univideo_alignment_images_1node1gpu.sbatch"):
        subprocess.run(["bash", "-n", str(shell_file)], check=True)
    env = dict(os.environ, PYTHON_BIN=sys.executable, HF_HOME=str(tmp_path), TMPDIR=str(tmp_path), DRY_RUN="0")
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Use sbatch/srun" in result.stderr


def test_index_reader_rejects_escape_and_corruption(tmp_path):
    with pytest.raises(ValueError):
        data.load_indexed_image(tmp_path, {"shard": "../escape"})
    (tmp_path / "a.tar").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="checksum"):
        data.load_indexed_image(tmp_path, {"shard": "a.tar", "image_offset": 0, "image_size": 3,
                                          "image_sha256": hashlib.sha256(b"good").hexdigest()})


@pytest.mark.parametrize("exit_code", [0, 23])
@pytest.mark.parametrize("scope", ["images", "stage1"])
def test_shell_propagates_download_status_and_stops_heartbeat(tmp_path, exit_code, scope):
    root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, signal, sys, time
root = pathlib.Path(os.environ['TEST_ROOT'])
if sys.argv[1] == '-c':
    sys.exit(0)
if sys.argv[1].endswith('gpu_heartbeat.py'):
    (root / 'heartbeat_started').touch()
    stop_file = pathlib.Path(sys.argv[sys.argv.index('--stop-file') + 1])
    def stop(*args):
        if args:
            (root / 'heartbeat_forced').touch()
        (root / 'heartbeat_stopped').touch()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        if stop_file.exists():
            stop()
        time.sleep(0.01)
(root / 'download_command.json').write_text(json.dumps(sys.argv))
time.sleep(0.1)
sys.exit(int(os.environ['TEST_EXIT']))
""")
    fake_python.chmod(0o755)
    srun = fake_bin / "srun"
    srun.write_text('#!/bin/sh\ncase " $* " in *" --cpu-bind=none "*) ;; *) exit 91 ;; esac\n'
                    'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
    sleep = fake_bin / "sleep"
    sleep.write_text('#!/bin/sh\n/bin/sleep 0.1\n')
    sleep.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", TEST_ROOT=str(tmp_path),
               TEST_EXIT=str(exit_code), PYTHON_BIN=str(fake_python), HF_HOME=str(tmp_path / "cache"),
               TMPDIR=str(tmp_path / "tmp"), DATASET_OUTPUT_DIR=str(tmp_path / "data"),
               SLURM_JOB_ID="unit-test-no-real-gpu", DRY_RUN="0", ALIGNMENT_SCOPE=scope)
    result = subprocess.run(["bash", str(root / "scripts/run_alignment_image_download.sh")],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr + result.stdout
    assert (tmp_path / "heartbeat_started").exists()
    assert (tmp_path / "heartbeat_stopped").exists()
    assert not (tmp_path / "heartbeat_forced").exists()
    assert not list((tmp_path / "tmp").glob("alignment-data-*"))
    command = json.loads((tmp_path / "download_command.json").read_text())
    expected = "prepare_stage1_alignment.py" if scope == "stage1" else "download_alignment_images.py"
    assert command[1].endswith(expected)


def test_rejection_gate_does_not_mark_dataset_complete(tmp_path, monkeypatch):
    args = data.parse_args(["--output-dir", str(tmp_path), "--num-shards", "1", "--disk-margin-gib", "0"])
    shard = data.Shard("sa_1.tar", 100, "a" * 64)
    monkeypatch.setattr(data, "resolve_plan", lambda *a, **kw: fixture_plan(shard, args))
    monkeypatch.setattr(data, "download_shard", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(data, "index_shard", lambda *a, **kw: {
        "valid_pairs": 50, "rejected": {"invalid_image_or_caption": 5}})
    with pytest.raises(RuntimeError, match="incomplete"):
        data.run(args)
    assert not (tmp_path / ".download_complete").exists()
    summary = json.loads((tmp_path / "download_summary.json").read_text())
    assert summary["reject_fraction"] > 0.01
