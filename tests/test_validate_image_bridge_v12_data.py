from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from new_mobile_ov.training.image_bridge_grounding_cascade import file_sha256


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_v12_data_preflight_accepts_disjoint_grounded_contract(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "green").save(image)
    write_csv(root / "manifests/d1_broad_train.csv", [{"record_id": "b", "caption": "A dog"}])
    write_csv(
        root / "manifests/d2_compositional_train.csv",
        [{"record_id": "c", "caption": "Two red cups"}],
    )
    grounded = {
        "record_id": "g",
        "caption": "A green image",
        "image_path": str(image),
        "capabilities": "color;attribute",
        "grounding_status": "qwen36_verified",
        "verification_source": "qwen36",
        "siglip_status": "",
        "siglip_logit": "",
    }
    write_csv(root / "manifests/d2_grounded_high_precision_50k.csv", [grounded])
    write_csv(root / "manifests/d2_grounded_candidate_100k.csv", [grounded])
    write_csv(root / "manifests/validation.csv", [{"record_id": "v", "caption": "A bird"}])
    write_csv(
        root / "manifests/hard_validation.csv",
        [{"record_id": "h", "caption": "Three blue balls"}],
    )
    mixture = {
        "overall_sampling": {"broad": 0.7, "compositional": 0.2, "grounded": 0.1},
        "trainer_contract": {
            "generation_source_names": ["broad", "compositional", "grounded_cascade"],
            "grounded_source_names": ["grounded_cascade"],
            "grounded_batch_probability": 0.1,
        },
    }
    mixture_path = root / "mixtures/v12_grounding_cascade_70_20_10.json"
    mixture_path.parent.mkdir(parents=True)
    mixture_path.write_text(json.dumps(mixture), encoding="utf-8")
    summary = {
        "outputs": {
            "candidate": {
                "sha256": file_sha256(root / "manifests/d2_grounded_candidate_100k.csv")
            },
            "high_precision": {
                "sha256": file_sha256(root / "manifests/d2_grounded_high_precision_50k.csv")
            },
            "mixture": {"sha256": file_sha256(mixture_path)},
        }
    }
    summary_path = root / "stats/grounding_cascade_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rescore = dict(grounded, siglip_logit="5.0")
    write_csv(root / "stats/rescore.csv", [rescore])

    subprocess.run(
        [
            sys.executable,
            "tools/validate_image_bridge_v12_data.py",
            "--data-root",
            str(root),
            "--output",
            "stats/preflight.json",
            "--rescore-manifest",
            str(root / "stats/rescore.csv"),
            "--expected-grounded-rows",
            "1",
            "--decode-samples",
            "1",
        ],
        check=True,
    )
    report = json.loads((root / "stats/preflight.json").read_text(encoding="utf-8"))
    assert report["machine_preflight_passed"] is True
    assert report["image_decode_check"]["status_counts"] == {"ok": 1}


def test_v12_data_preflight_rejects_pool_overlap(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "blue").save(image)
    common = {"record_id": "same", "caption": "A blue square"}
    write_csv(root / "manifests/d1_broad_train.csv", [common])
    write_csv(root / "manifests/d2_compositional_train.csv", [common])
    grounded = {
        **common,
        "record_id": "g",
        "image_path": str(image),
        "capabilities": "color",
        "grounding_status": "siglip2_high_confidence",
        "verification_source": "siglip2",
        "siglip_status": "ok",
        "siglip_logit": "4.0",
    }
    write_csv(root / "manifests/d2_grounded_high_precision_50k.csv", [grounded])
    write_csv(root / "manifests/d2_grounded_candidate_100k.csv", [grounded])
    write_csv(root / "manifests/validation.csv", [{"record_id": "v", "caption": "A bird"}])
    write_csv(root / "manifests/hard_validation.csv", [{"record_id": "h", "caption": "A cat"}])
    mixture_path = root / "mixtures/v12_grounding_cascade_70_20_10.json"
    mixture_path.parent.mkdir(parents=True)
    mixture_path.write_text(
        json.dumps(
            {
                "overall_sampling": {"broad": 0.7, "compositional": 0.2, "grounded": 0.1},
                "trainer_contract": {
                    "generation_source_names": ["broad", "compositional", "grounded_cascade"],
                    "grounded_source_names": ["grounded_cascade"],
                    "grounded_batch_probability": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    summary_path = root / "stats/grounding_cascade_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "tools/validate_image_bridge_v12_data.py",
            "--data-root",
            str(root),
            "--expected-grounded-rows",
            "1",
            "--decode-samples",
            "1",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "overlap.broad__compositional=1" in result.stderr
