from __future__ import annotations

import csv

from PIL import Image

from new_mobile_ov.training.grounded_manifest import filter_grounded_manifest
from new_mobile_ov.training.prompt_curriculum import CaptionManifestDataset


def test_filter_grounded_manifest_keeps_only_verified_images(tmp_path) -> None:
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), "orange").save(image_path)
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image")
    source = tmp_path / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["caption", "image_path"])
        writer.writeheader()
        writer.writerows(
            [
                {"caption": "valid", "image_path": str(image_path)},
                {"caption": "missing", "image_path": str(tmp_path / "missing.jpg")},
                {"caption": "corrupt", "image_path": str(corrupt_path)},
            ]
        )
    output = tmp_path / "filtered.csv"

    summary = filter_grounded_manifest(
        source_manifest=source,
        output_manifest=output,
        source_name="shortcaption",
        image_columns=["image_path"],
        workers=2,
        progress_every=0,
    )

    assert summary.scanned_rows == 3
    assert summary.accepted_rows == 1
    assert summary.missing_image_rows == 1
    assert summary.unreadable_image_rows == 1
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "caption": "valid",
            "image_path": str(image_path),
            "source_name": "shortcaption",
        }
    ]
    dataset = CaptionManifestDataset(
        output,
        source_name="shortcaption_verified",
        variant_columns=[],
        variant_weights=[],
        fallback_column="caption",
        image_columns=["image_path"],
        max_samples=-1,
        require_existing_image=True,
    )
    assert dataset[0].image_path == str(image_path)
