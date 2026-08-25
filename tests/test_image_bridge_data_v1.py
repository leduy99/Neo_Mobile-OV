from __future__ import annotations

import csv
import json
import sqlite3

from PIL import Image

from new_mobile_ov.training.image_bridge_data_v1 import (
    ImageBridgeBuildConfig,
    ImageBridgeSource,
    build_image_bridge_data_v1,
    infer_capabilities,
)
from tools.verify_image_bridge_grounding_qwen36 import (
    extract_json,
    validate_annotation,
)


def write_manifest(path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_image_bridge_data_builder_creates_disjoint_provenance_preserving_views(
    tmp_path,
) -> None:
    dog_image = tmp_path / "dog.png"
    kitchen_image = tmp_path / "kitchen.png"
    Image.new("RGB", (8, 8), "brown").save(dog_image)
    Image.new("RGB", (8, 8), "white").save(kitchen_image)

    broad = tmp_path / "broad.csv"
    compositional = tmp_path / "compositional.csv"
    grounded = tmp_path / "grounded.csv"
    write_manifest(
        broad,
        [
            {"id": "b0", "caption": "A dog in a park"},
            {"id": "b1", "caption": "Two red cups are left of a blue bowl."},
            {"id": "b2", "caption": "A quiet lake."},
            {"id": "b3", "caption": "A DOG IN A PARK."},
        ],
    )
    write_manifest(
        compositional,
        [
            {
                "id": "c0",
                "caption": "Three yellow birds are above a black car.",
            }
        ],
    )
    write_manifest(
        grounded,
        [
            {
                "id": "g0",
                "caption": "A dog in a park",
                "image_path": str(dog_image),
            },
            {
                "id": "g1",
                "caption": "A woman walking in a kitchen.",
                "image_path": str(kitchen_image),
            },
        ],
    )

    output = tmp_path / "image_bridge_v1"
    summary = build_image_bridge_data_v1(
        [
            ImageBridgeSource("broad_source", "broad", broad),
            ImageBridgeSource("composition_source", "compositional", compositional),
            ImageBridgeSource("grounded_source", "grounded", grounded),
        ],
        ImageBridgeBuildConfig(
            output_dir=output,
            seed=17,
            validation_fraction=0.0,
            hard_validation_per_capability=0,
            max_broad_records=-1,
            max_compositional_records=-1,
            max_grounded_records=-1,
            source_hashes=False,
        ),
        progress_every=0,
    )

    broad_rows = read_manifest(output / "manifests" / "d1_broad_train.csv")
    composition_rows = read_manifest(
        output / "manifests" / "d2_compositional_train.csv"
    )
    grounded_rows = read_manifest(
        output / "manifests" / "d2_grounded_candidates.csv"
    )
    pools = [broad_rows, composition_rows, grounded_rows]
    ids = [{row["record_id"] for row in rows} for rows in pools]

    assert summary["unique_records"] == 5
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert {row["caption"] for row in broad_rows} == {"A quiet lake."}
    assert {row["caption"] for row in composition_rows} == {
        "Two red cups are left of a blue bowl.",
        "Three yellow birds are above a black car.",
    }
    assert {row["caption"] for row in grounded_rows} == {
        "A dog in a park",
        "A woman walking in a kitchen.",
    }

    dog = next(row for row in grounded_rows if row["caption"] == "A dog in a park")
    assert dog["source_name"] == "grounded_source"
    assert dog["source_key"] == "g0"
    assert dog["source_names"] == "broad_source;grounded_source"
    assert dog["source_role"] == "grounded"
    assert dog["image_path"] == str(dog_image)
    assert dog["grounding_status"] == "path_verified"

    mixture = json.loads(
        (output / "mixtures" / "v12_70_20_10.json").read_text(encoding="utf-8")
    )
    assert mixture["overall_sampling"] == {
        "broad": 0.70,
        "compositional": 0.20,
        "grounded": 0.10,
    }
    assert mixture["trainer_contract"]["generation_source_weights"] == [
        7.0 / 9.0,
        2.0 / 9.0,
        0.0,
    ]
    assert mixture["trainer_contract"]["semantic_prompt_probability"] == 0.0


def test_builder_seed_controls_selection_key(tmp_path) -> None:
    source = tmp_path / "source.csv"
    write_manifest(source, [{"caption": "A blue boat on a lake."}])
    keys = []
    for seed in (7, 19):
        output = tmp_path / f"seed_{seed}"
        build_image_bridge_data_v1(
            [ImageBridgeSource("source", "broad", source)],
            ImageBridgeBuildConfig(
                output_dir=output,
                seed=seed,
                validation_fraction=0.0,
                hard_validation_per_capability=0,
                max_broad_records=-1,
                max_compositional_records=0,
                max_grounded_records=0,
                source_hashes=False,
            ),
            progress_every=0,
        )
        with sqlite3.connect(output / "catalog.sqlite3") as connection:
            keys.append(connection.execute("SELECT selection_key FROM records").fetchone()[0])
    assert keys[0] != keys[1]


def test_capability_mining_detects_binding_count_and_relation() -> None:
    capabilities = set(
        infer_capabilities("Two red cups are left of a blue bowl in a kitchen.")
    )
    assert {
        "attribute_binding",
        "color",
        "count",
        "multiple_objects",
        "scene",
        "spatial_relation",
    }.issubset(capabilities)


def test_qwen_annotation_parser_accepts_fenced_json_and_rejects_bad_types() -> None:
    payload = extract_json(
        """```json
        {
          "caption_supported": true,
          "confidence": 0.93,
          "objects": ["dog"],
          "counts": [{"object": "dog", "count": 1}],
          "attributes": [],
          "relations": [],
          "scene": "park",
          "action": "",
          "reason": "The dog and park are visible."
        }
        ```"""
    )
    annotation = validate_annotation(payload)
    assert annotation["caption_supported"] is True
    assert annotation["confidence"] == 0.93

    payload["objects"] = "dog"
    try:
        validate_annotation(payload)
    except ValueError as exc:
        assert "objects must be a list" in str(exc)
    else:
        raise AssertionError("A non-list objects field must be rejected")
