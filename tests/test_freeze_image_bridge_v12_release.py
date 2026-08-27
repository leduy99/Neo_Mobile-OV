from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["record_id"] for row in csv.DictReader(handle)]


def test_freeze_v12_release_filters_leakage_without_mutating_source(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_rows = [
        {"record_id": "keep", "caption": "A green bird on a branch"},
        {"record_id": "drop", "caption": "A dog runs through snow"},
    ]
    write_csv(data_root / "manifests/d1_broad_train.csv", source_rows)
    write_csv(
        data_root / "manifests/d2_compositional_train.csv",
        [{"record_id": "comp", "caption": "Two red cups beside a bowl"}],
    )
    write_csv(
        data_root / "manifests/d2_grounded_high_precision_50k.csv",
        [{"record_id": "ground", "caption": "A blue car in a city"}],
    )
    write_csv(data_root / "manifests/validation.csv", [{"record_id": "v", "caption": "A cat"}])
    write_csv(
        data_root / "manifests/hard_validation.csv",
        [{"record_id": "h", "caption": "Three yellow balls"}],
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A dog runs through snow\n", encoding="utf-8")
    release = data_root / "releases/v12"
    command = [
        sys.executable,
        "tools/freeze_image_bridge_v12_release.py",
        "--data-root",
        str(data_root),
        "--output-root",
        str(release),
        "--leakage-prompts",
        str(prompts),
    ]
    subprocess.run(command, check=True)
    assert read_ids(data_root / "manifests/d1_broad_train.csv") == ["keep", "drop"]
    assert read_ids(release / "manifests/d1_broad_train.csv") == ["keep"]
    summary = json.loads((release / "stats/release_summary.json").read_text())
    assert summary["manifests"]["broad"]["filtered_rows"] == 1
    assert Path(summary["manifests"]["broad"]["output"]) == (
        release / "manifests/d1_broad_train.csv"
    )

    rerun = subprocess.run(command, check=True, text=True, capture_output=True)
    assert '"reused_existing_release": true' in rerun.stdout
