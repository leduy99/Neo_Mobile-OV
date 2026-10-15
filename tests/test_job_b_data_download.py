import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import download_mobileov_job_b_data as download


def file_spec(filename, payload, kind, algorithm="sha256"):
    digest = hashlib.sha256(payload).hexdigest() if algorithm == "sha256" else hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return dict(filename=filename, size=len(payload), kind=kind,
                hash_algorithm=algorithm, hash=digest)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    payloads = {}
    plan = dict(format=download.FORMAT, sources={})
    for name, filename, kind in (
        ("image_qa", "vqa/train-00000.parquet", "train_parquet"),
        ("text_instruction", "data/train-00000.parquet", "train_parquet"),
        ("video_qa", "videos/media_1.tar.gz", "media_archive"),
    ):
        body, card = f"selected data {name}".encode(), f"source card {name}".encode()
        payloads[name, filename] = body
        payloads[name, "README.md"] = card
        plan["sources"][name] = dict(repo_id=f"fixture/{name}", revision="a" * 40,
                                    files=[file_spec(filename, body, kind),
                                           file_spec("README.md", card, "dataset_card", "git_blob_sha1")])
    annotation = b'[{"video":"media.mp4","conversations":[]}]'
    payloads["video_qa", "videos/qa.json"] = annotation
    plan["sources"]["video_qa"]["files"].append(file_spec("videos/qa.json", annotation, "qa_annotation"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args = download.parse_args(["--plan", str(plan_path), "--output-dir", str(tmp_path / "out"),
                                "--disk-margin-gib", "0"])
    metadata_calls, fetches = [], []

    class API:
        def dataset_info(self, repo_id, revision, files_metadata):
            metadata_calls.append((repo_id, revision, files_metadata))
            source = next(s for s in plan["sources"].values() if s["repo_id"] == repo_id)
            siblings = [SimpleNamespace(rfilename=f["filename"], size=f["size"],
                        lfs=dict(sha256=f["hash"]) if f["hash_algorithm"] == "sha256" else None,
                        blob_id=f["hash"] if f["hash_algorithm"] == "git_blob_sha1" else None)
                        for f in source["files"]]
            return SimpleNamespace(sha=source["revision"], siblings=siblings)

    def fetch(**kwargs):
        fetches.append(kwargs)
        name = kwargs["repo_id"].split("/")[-1]
        assert kwargs["repo_type"] == "dataset" and kwargs["revision"] == "a" * 40
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[name, kwargs["filename"]])
        return str(path)

    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024**4))
    return SimpleNamespace(args=args, plan=plan, api=API(), fetch=fetch, payloads=payloads,
                           fetches=fetches, metadata_calls=metadata_calls)


def execute(corpus, **kwargs):
    return download.run(corpus.args, api=corpus.api, fetch=kwargs.pop("fetch", corpus.fetch),
                        sleep=lambda _: None, **kwargs)


def test_real_plan_is_bounded_and_explicit():
    plan = json.loads(download.DEFAULT_PLAN.read_text())
    rows = download.selected_files(plan)
    assert len(rows) == 14
    assert sum(row["kind"] == "dataset_card" for row in rows) == 3
    assert 8 < sum(row["size"] for row in rows) / 1024**3 < 9
    assert {row["repo_id"] for row in rows} == {
        "HuggingFaceM4/the_cauldron", "HuggingFaceTB/smol-smoltalk", "lmms-lab/LLaVA-Video-178K"}
    assert {row["filename"].split("/")[0] for row in rows if row["source"] == "image_qa"} == {
        "vqav2", "textvqa", "ai2d", "chartqa", "clevr", "tallyqa", "vsr", "README.md"}
    assert "no_KL_requirement" in plan["purpose"]


def test_dry_run_has_no_io_or_network(corpus):
    corpus.args.dry_run = True
    report = execute(corpus)
    assert not corpus.metadata_calls and not corpus.fetches and not corpus.args.output_dir.exists()
    assert report["selected_files"] == 7 and not report["trainer_ready"]


def test_metadata_only_does_not_claim_source_completion(corpus):
    corpus.args.metadata_only = True
    report = execute(corpus)
    assert len(corpus.metadata_calls) == 3 and not corpus.fetches
    assert report["status"] == "Hub_metadata_verified_NO_source_download"
    assert (corpus.args.output_dir / "source_contract.json").exists()
    assert not (corpus.args.output_dir / "source_files_complete.json").exists()


def test_download_is_pinned_parallel_and_resumable(corpus):
    report = execute(corpus)
    assert len(corpus.fetches) == 7 and len(report["files"]) == 7
    assert not report["trainer_ready"] and not report["original_splits_verified"]
    assert not report["vae_encoded"] and report["unique_qa_counts"] is None
    assert len(list((corpus.args.output_dir / "receipts").glob("*.json"))) == 7
    assert all(row["source_file_verified"] for row in report["files"])
    assert all(Path(f["local_dir"]).name == f["repo_id"].split("/")[-1] for f in corpus.fetches)
    corpus.fetches.clear()
    execute(corpus)
    assert not corpus.fetches


def test_same_size_corrupt_cache_is_refetched(corpus):
    report = execute(corpus)
    row = report["files"][0]
    Path(row["path"]).write_bytes(b"x" * row["size"])
    corpus.fetches.clear()
    execute(corpus)
    assert len(corpus.fetches) == 1 and corpus.fetches[0]["force_download"]
    download.verify_file(Path(row["path"]), row)


def test_retry_after_partial_download(corpus):
    failures = []

    def interrupted(**kwargs):
        if kwargs["repo_id"] == "fixture/video_qa" and kwargs["filename"].endswith(".tar.gz") and not failures:
            failures.append(True)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
            raise OSError("interrupted connection")
        return corpus.fetch(**kwargs)

    execute(corpus, fetch=interrupted)
    assert failures
    assert any(f["force_download"] and f["filename"].endswith(".tar.gz") for f in corpus.fetches)


def test_persistent_corruption_never_leaves_completion_marker(corpus):
    execute(corpus)
    bad = corpus.args.output_dir / "sources/video_qa/videos/media_1.tar.gz"
    bad.write_bytes(b"corrupt")
    attempts = []

    def corrupt(**kwargs):
        if kwargs["filename"].endswith(".tar.gz"):
            attempts.append(kwargs)
            return str(bad)
        return corpus.fetch(**kwargs)

    with pytest.raises(RuntimeError, match="Failed source video_qa"):
        execute(corpus, fetch=corrupt)
    assert len(attempts) == corpus.args.retries
    assert not (corpus.args.output_dir / "source_files_complete.json").exists()


def test_git_blob_digest_includes_header(corpus, tmp_path):
    row = dict(corpus.plan["sources"]["image_qa"]["files"][1], source="image_qa")
    path = tmp_path / "card.md"
    path.write_bytes(corpus.payloads["image_qa", "README.md"])
    download.verify_file(path, row)
    plain_hash = dict(row, hash=hashlib.sha1(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        download.verify_file(path, plain_hash)


@pytest.mark.parametrize("fault", ["revision", "size", "filename", "digest"])
def test_hub_metadata_mismatch_fails_before_download(corpus, fault):
    original = corpus.api.dataset_info

    def changed(*args, **kwargs):
        info = original(*args, **kwargs)
        if fault == "revision":
            info.sha = "b" * 40
        elif fault == "size":
            info.siblings[0].size += 1
        elif fault == "filename":
            info.siblings[0].rfilename = "missing"
        else:
            info.siblings[0].lfs = dict(sha256="0" * 64)
        return info

    corpus.api.dataset_info = changed
    with pytest.raises(ValueError, match="Hub"):
        execute(corpus)
    assert not corpus.fetches and not corpus.args.output_dir.exists()


def test_budget_limit_fails_before_network(corpus):
    corpus.args.max_download_gib = 1e-12
    with pytest.raises(ValueError, match="storage cap"):
        execute(corpus)
    assert not corpus.fetches and not corpus.metadata_calls


def test_low_disk_invalidates_old_completion(corpus, monkeypatch):
    execute(corpus)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Insufficient free space"):
        execute(corpus)
    assert not (corpus.args.output_dir / "source_files_complete.json").exists()


def test_changed_contract_requires_new_output(corpus):
    execute(corpus)
    marker = corpus.args.output_dir / "source_files_complete.json"
    original = marker.read_bytes()
    corpus.plan["purpose"] = "changed selection policy"
    corpus.args.plan.write_text(json.dumps(corpus.plan))
    corpus.fetches.clear()
    with pytest.raises(ValueError, match="selection changed"):
        execute(corpus)
    assert not corpus.fetches and marker.read_bytes() == original


def test_duplicate_downloader_is_rejected(corpus):
    with download.output_lock(corpus.args.output_dir):
        with pytest.raises(RuntimeError, match="Another downloader"):
            execute(corpus)
    assert not corpus.fetches


def test_destination_symlink_cannot_escape(corpus, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = corpus.args.output_dir
    root.mkdir()
    (root / "sources").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        execute(corpus)
    assert not corpus.fetches and not list(outside.iterdir())


@pytest.mark.parametrize("filename", ["../file", "/file", "foo//file", "foo\\file", ".", "foo\0bar", "foo\nbar"])
def test_unsafe_source_paths_rejected(corpus, filename):
    plan = copy.deepcopy(corpus.plan)
    plan["sources"]["image_qa"]["files"][0]["filename"] = filename
    with pytest.raises(ValueError):
        download.selected_files(plan)


@pytest.mark.parametrize("mutation", ["moving_revision", "duplicate", "test_split", "bad_hash", "card_only", "missing_source"])
def test_invalid_selection_rejected(corpus, mutation):
    plan = copy.deepcopy(corpus.plan)
    source = plan["sources"]["image_qa"]
    if mutation == "moving_revision":
        source["revision"] = "main"
    elif mutation == "duplicate":
        source["files"].append(source["files"][0])
    elif mutation == "test_split":
        source["files"][0]["filename"] = "vqa/test-00000.parquet"
    elif mutation == "bad_hash":
        source["files"][0]["hash"] = "bad"
    elif mutation == "card_only":
        source["files"] = source["files"][1:]
    else:
        del plan["sources"]["video_qa"]
    with pytest.raises(ValueError):
        download.selected_files(plan)


@pytest.mark.parametrize("argv", [["--workers", "0"], ["--workers", "17"], ["--retries", "0"],
                                  ["--max-download-gib", "nan"], ["--max-download-gib", "inf"],
                                  ["--disk-margin-gib", "-1"], ["--disk-margin-gib", "nan"],
                                  ["--dry-run", "--metadata-only"]])
def test_invalid_cli_options(argv):
    with pytest.raises(SystemExit):
        download.parse_args(argv)


def test_bulk_download_requires_slurm(corpus, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="CPU Slurm job"):
        download.main(["--plan", str(corpus.args.plan)])


@pytest.mark.parametrize("exit_code", [0, 7])
def test_cpu_launcher_command_and_exit_status(tmp_path, exit_code):
    launcher = download.ROOT / "scripts/download_mobileov_job_b_data_1nodecpu.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = launcher.read_text()
    assert "#SBATCH --partition=berzelius-cpu" in source
    assert "--gres" not in source and "heartbeat" not in source
    assert "train_mobileov" not in source and "sbatch " not in source
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "args.json"
    mock = bin_dir / "srun"
    mock.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
                    'sys.exit(int(os.environ["MOCK_EXIT"]))\n')
    mock.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", PROJECT_ROOT=str(tmp_path),
               PYTHON="/bin/true", CAPTURE=str(capture), MOCK_EXIT=str(exit_code),
               JOB_B_DATA_ROOT="/custom/qa sources", DOWNLOAD_WORKERS="3", MAX_DOWNLOAD_GIB="9")
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == exit_code
    argv = json.loads(capture.read_text())
    assert argv[:5] == ["--ntasks=1", "/bin/true", "-s", "-u", "tools/download_mobileov_job_b_data.py"]
    assert argv[argv.index("--plan") + 1] == "configs/data_sources/mobileov_job_b_understanding_sources.json"
    assert argv[argv.index("--output-dir") + 1] == "/custom/qa sources"
    assert argv[argv.index("--workers") + 1] == "3"
