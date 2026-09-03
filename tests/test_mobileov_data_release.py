from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from new_mobile_ov.training.mobileov_data_release import (
    ReleaseBuildConfig,
    ReleaseSource,
    build_mobileov_data_release,
)
from new_mobile_ov.training.mobileov_prompt_bank import (
    PromptSource,
    build_anchor_prompt_bank,
)
from tools.prepare_mobileov_anchor_teacher_data import valid_trajectory
from tools.score_mobileov_video_data import contact_sheet, motion_metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_latent(path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": torch.zeros(16, 7, 4, 4), **extra}, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_release_builder_preserves_pool_contracts_and_filters_benchmark(tmp_path) -> None:
    image = tmp_path / "image.png"
    anchor = tmp_path / "matched" / "anchors" / "anchor.png"
    Image.new("RGB", (16, 16), "red").save(image)
    anchor.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(anchor)

    real_latent = tmp_path / "real" / "latents" / "real.pt"
    teacher_latent = tmp_path / "teacher" / "latents" / "teacher.pt"
    matched_latent = tmp_path / "matched" / "latents" / "matched.pt"
    save_latent(real_latent)
    save_latent(teacher_latent)
    save_latent(
        matched_latent,
        pair_contract="same_dreamlite_anchor_native_teacher",
        anchor_image_path="anchors/anchor.png",
        anchor_sha256=sha256(anchor),
    )

    source_rows = {
        "broad": [{"id": "b0", "caption": "A quiet illustrated forest."}],
        "composition": [
            {"id": "c0", "caption": "Two red birds fly above a blue car."},
            {"id": "leak", "caption": "A benchmark-only dancing robot."},
        ],
        "grounded": [
            {"id": "g0", "caption": "A brown dog runs outside.", "image_path": str(image)}
        ],
        "real": [
            {
                "sample_id": "r0",
                "prompt": "A cyclist turns on a wet road.",
                "latent_path": "latents/real.pt",
            }
        ],
        "teacher": [
            {
                "index": "t0",
                "prompt": "A paper boat floats down a stream.",
                "latent_path": "latents/teacher.pt",
            }
        ],
        "matched": [
            {
                "sample_id": "m0",
                "prompt": "A fox walks through fresh snow.",
                "latent_path": "latents/matched.pt",
                "anchor_image_path": "anchors/anchor.png",
            }
        ],
    }
    source_paths = {}
    for name, rows in source_rows.items():
        directory = tmp_path / name
        if name in {"real", "teacher", "matched"}:
            directory = tmp_path / name
        path = directory / "manifest.csv"
        write_csv(path, rows)
        source_paths[name] = path
    benchmark = tmp_path / "benchmark.txt"
    benchmark.write_text("A benchmark-only dancing robot.\n", encoding="utf-8")

    output = tmp_path / "release"
    summary = build_mobileov_data_release(
        [
            ReleaseSource("broad", "image_broad", source_paths["broad"], require_artifact=False),
            ReleaseSource("composition", "image_compositional", source_paths["composition"], require_artifact=False),
            ReleaseSource("grounded", "image_grounded", source_paths["grounded"]),
            ReleaseSource("real", "video_real", source_paths["real"]),
            ReleaseSource("teacher", "video_teacher_t2v", source_paths["teacher"]),
            ReleaseSource("matched", "video_anchor_teacher", source_paths["matched"]),
        ],
        ReleaseBuildConfig(
            output_dir=output,
            validation_fraction=0.0,
            source_hashes=False,
        ),
        benchmark_prompt_paths=[benchmark],
        progress_every=0,
    )

    assert summary["source_counts"]["composition"]["benchmark_leakage"] == 1
    assert all(summary["pool_counts"][f"{pool}_train"] == 1 for pool in (
        "image_broad",
        "image_compositional",
        "image_grounded",
        "video_real",
        "video_teacher_t2v",
        "video_anchor_teacher",
    ))
    matched = read_csv(output / "manifests/video_anchor_teacher_train.csv")[0]
    assert Path(matched["latent_path"]) == matched_latent.resolve()
    assert Path(matched["anchor_image_path"]) == anchor.resolve()
    assert matched["task"] == "video_anchor_conditioned_teacher_distill"
    assert json.loads((output / "release.json").read_text())["schema_version"] == 1
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/validate_mobileov_data_release.py"),
            "--release-root",
            str(output),
            "--sample-checks-per-pool",
            "2",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
    )
    assert json.loads((output / "stats/preflight.json").read_text())["passed"] is True


def test_prompt_bank_is_deterministic_balanced_and_unique(tmp_path) -> None:
    video = tmp_path / "video.csv"
    image = tmp_path / "image.csv"
    write_csv(
        video,
        [
            {"id": f"v{i}", "caption_long": f"A person walks through scene number {i}."}
            for i in range(8)
        ],
    )
    write_csv(
        image,
        [
            {"id": f"i{i}", "caption": f"Two red birds fly above a blue car number {i}."}
            for i in range(8)
        ],
    )
    output = tmp_path / "prompts.csv"
    summary = build_anchor_prompt_bank(
        [
            PromptSource("video", video, weight=0.75, hard_fraction=0.0),
            PromptSource("image", image, weight=0.25, hard_fraction=1.0),
        ],
        output_path=output,
        target_records=8,
        seed=7,
        progress_every=0,
    )
    rows = read_csv(output)
    assert len(rows) == 8
    assert len({row["prompt_id"] for row in rows}) == 8
    assert summary["source_selected"] == {"video": 6, "image": 2}
    assert sum(row["source_name"] == "image" for row in rows) == 2


def test_motion_metrics_distinguish_static_and_moving_frames() -> None:
    static = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(4)]
    moving = []
    for offset in (4, 16, 28, 40):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[20:40, offset : offset + 20] = 255
        moving.append(frame)
    static_metrics = motion_metrics(static)
    moving_metrics = motion_metrics(moving)
    assert static_metrics["motion_frame_diff_mean"] == 0.0
    assert moving_metrics["motion_frame_diff_mean"] > 0.01
    assert moving_metrics["motion_optical_flow_mean"] > static_metrics["motion_optical_flow_mean"]
    assert contact_sheet(moving).size == (768, 288)


def test_video_finalizer_rejects_static_and_misaligned_rows(tmp_path) -> None:
    scores = tmp_path / "scores.csv"
    rows = []
    for index in range(20):
        rows.append(
            {
                "sample_id": index,
                "prompt": f"A moving subject {index}",
                "video_score_status": "ok",
                "siglip_logit": index,
                "motion_frame_diff_mean": index / 1000,
                "motion_optical_flow_mean": index / 100,
                "transition_max_diff": 0.05 if index < 19 else 0.8,
            }
        )
    write_csv(scores, rows)
    output = tmp_path / "selected.csv"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/finalize_mobileov_video_data.py"),
            "--scores",
            str(scores),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
    )
    selected = read_csv(output)
    assert selected
    assert all(float(row["motion_frame_diff_mean"]) >= 0.005 for row in selected)
    assert all(float(row["motion_optical_flow_mean"]) >= 0.05 for row in selected)
    assert all(float(row["transition_max_diff"]) <= 0.35 for row in selected)
    assert all(row["verification_status"] == "video_cascade_pass" for row in selected)


def test_matched_trajectory_contract_is_machine_checkable(tmp_path) -> None:
    good = tmp_path / "good.pt"
    bad = tmp_path / "bad.pt"
    save_latent(good, pair_contract="same_dreamlite_anchor_native_teacher")
    save_latent(bad, pair_contract="native_t2v")
    assert valid_trajectory(good)
    assert not valid_trajectory(bad)
