from __future__ import annotations

from collections import Counter

import pandas as pd
from PIL import Image

from new_mobile_ov.training.prompt_curriculum import (
    CaptionManifestDataset,
    MixedPromptDataset,
    prompt_example_collate,
)


def manifest_dataset(tmp_path, name: str, rows: list[dict]) -> CaptionManifestDataset:
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return CaptionManifestDataset(
        path,
        source_name=name,
        variant_columns=["caption_short", "caption_medium", "caption_long"],
        variant_weights=[1.0, 1.0, 1.0],
        fallback_column="caption",
        image_columns=["image_path", "media_path", "video_path"],
        max_samples=-1,
    )


def test_manifest_dataset_preserves_caption_source_and_raw_image(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "red").save(image_path)
    dataset = manifest_dataset(
        tmp_path,
        "mobileo",
        [
            {
                "caption": "fallback",
                "caption_short": "a red cube",
                "image_path": str(image_path),
            }
        ],
    )

    example = dataset[0]
    assert example.prompt == "a red cube"
    assert example.source == "mobileo"
    assert example.image_path == str(image_path)
    assert dataset.image_candidate_rows == 1


def test_mixed_prompt_dataset_uses_source_weights_not_source_sizes(tmp_path) -> None:
    major = manifest_dataset(
        tmp_path,
        "major",
        [{"caption": f"major {index}"} for index in range(3)],
    )
    minor = manifest_dataset(
        tmp_path,
        "minor",
        [{"caption": "minor"}],
    )
    mixed = MixedPromptDataset([major, minor], [0.8, 0.2], seed=17)

    counts = Counter(mixed[index].source for index in range(2000))
    major_fraction = counts["major"] / sum(counts.values())
    assert 0.76 < major_fraction < 0.84
    assert mixed.source_summary[0]["weight"] == 0.8
    assert mixed.source_summary[0]["image_candidate_rows"] == 0


def test_prompt_collate_keeps_optional_images_aligned(tmp_path) -> None:
    dataset = manifest_dataset(
        tmp_path,
        "source",
        [{"caption": "one"}, {"caption": "two"}],
    )
    prompts, image_paths, sources = prompt_example_collate([dataset[0], dataset[1]])

    assert prompts == ["one", "two"]
    assert image_paths == ["", ""]
    assert sources == ["source", "source"]


def test_train_ready_video_path_keeps_images_but_rejects_video_files(tmp_path) -> None:
    image_path = tmp_path / "mobile_o.jpg"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    dataset = manifest_dataset(
        tmp_path,
        "train_ready",
        [
            {"caption": "an image row", "video_path": str(image_path)},
            {"caption": "a video row", "video_path": str(tmp_path / "clip.mp4")},
        ],
    )

    assert dataset[0].image_path == str(image_path)
    assert dataset[1].image_path == ""
    assert dataset.image_candidate_rows == 1
