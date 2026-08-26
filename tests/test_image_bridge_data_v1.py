from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from new_mobile_ov.training.image_bridge_data_v1 import (
    ImageBridgeBuildConfig,
    ImageBridgeSource,
    build_image_bridge_data_v1,
    infer_capabilities,
)
from new_mobile_ov.training.image_bridge_grounding_cascade import (
    LeakageIndex,
    balanced_select,
    capability_bucket,
    qwen_decisions,
)
from tools.verify_image_bridge_grounding_qwen36 import (
    QwenVisualVerifier,
    extract_json,
    ensure_csv_fields,
    load_csv_ids,
    load_processed,
    merge_csv_records,
    merge_jsonl_records,
    validate_annotation,
    verification_instruction,
)
from tools.score_image_bridge_grounding_siglip import score_batch


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


def test_qwen_full_run_bootstrap_reuses_pilot_outputs(tmp_path) -> None:
    pilot_jsonl = tmp_path / "pilot.jsonl"
    full_jsonl = tmp_path / "full.jsonl"
    pilot_jsonl.write_text(
        '\n'.join(
            [
                json.dumps({"record_id": "a", "accepted": True}),
                json.dumps({"record_id": "b", "accepted": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    full_jsonl.write_text(
        json.dumps({"record_id": "a", "accepted": True}) + "\n",
        encoding="utf-8",
    )
    assert merge_jsonl_records(pilot_jsonl, full_jsonl) == 1
    assert merge_jsonl_records(pilot_jsonl, full_jsonl) == 0
    assert load_processed(full_jsonl) == {"a", "b"}

    pilot_csv = tmp_path / "pilot.csv"
    full_csv = tmp_path / "full.csv"
    write_manifest(
        pilot_csv,
        [
            {"record_id": "a", "caption": "A dog."},
            {"record_id": "c", "caption": "A cat."},
        ],
    )
    write_manifest(full_csv, [{"record_id": "a", "caption": "A dog."}])
    assert merge_csv_records(pilot_csv, full_csv, ["record_id", "caption"]) == 1
    assert merge_csv_records(pilot_csv, full_csv, ["record_id", "caption"]) == 0
    assert load_csv_ids(full_csv) == {"a", "c"}


def test_qwen_verifier_batches_images_and_isolates_bad_files(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 16), "blue").save(image_path)
    verifier = QwenVisualVerifier.__new__(QwenVisualVerifier)
    verifier.max_image_edge = 16
    verifier._generate_with_oom_split = lambda images, captions: [
        json.dumps(
            {
                "caption_supported": True,
                "confidence": 0.9,
                "objects": ["boat"],
                "counts": [],
                "attributes": [],
                "relations": [],
                "scene": "lake",
                "action": "",
                "reason": "Visible.",
            }
        )
        for _ in images
    ]
    results = verifier.verify_batch(
        [(str(image_path), "A boat."), (str(tmp_path / "missing.png"), "Missing.")]
    )
    assert results[0][0]["caption_supported"] is True
    assert results[0][2] is None
    assert results[1][0] is None
    assert "FileNotFoundError" in results[1][2]


def test_compact_qwen_schema_and_retryable_latest_decision(tmp_path) -> None:
    annotation = validate_annotation(
        {
            "caption_supported": False,
            "confidence": 0.91,
            "failed_claims": ["three dogs"],
        }
    )
    assert annotation["objects"] == []
    assert annotation["failed_claims"] == ["three dogs"]
    instruction = verification_instruction("Three dogs.", schema="compact")
    assert "failed_claims" in instruction
    assert "objects" not in instruction

    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"record_id": "retry", "error": "parse"}),
                json.dumps({"record_id": "accepted", "accepted": True}),
                json.dumps({"record_id": "retry", "accepted": False}),
                json.dumps({"record_id": "latest_error", "accepted": True}),
                json.dumps({"record_id": "latest_error", "error": "oom"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert qwen_decisions(annotations) == {
        "retry": "rejected",
        "accepted": "accepted",
        "latest_error": "error",
    }
    assert load_processed(annotations, retry_errors=True) == {"retry", "accepted"}


def test_accepted_manifest_schema_is_upgraded_for_resume(tmp_path) -> None:
    manifest = tmp_path / "accepted.csv"
    write_manifest(manifest, [{"record_id": "a", "caption": "A dog."}])
    ensure_csv_fields(manifest, ["record_id", "caption", "qwen36_failed_claims"])
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "qwen36_failed_claims" in (reader.fieldnames or [])
        assert list(reader)[0]["record_id"] == "a"


def test_grounding_cascade_balances_deduplicates_and_blocks_leakage() -> None:
    rows = [
        {
            "record_id": f"r{index}",
            "image_path": f"/{index}.jpg",
            "caption": caption,
            "capabilities": capabilities,
            "siglip_score": score,
        }
        for index, (caption, capabilities, score) in enumerate(
            [
                ("A dog in a city park.", "scene", "0.99"),
                ("Two red cups left of a bowl.", "count;spatial_relation", "0.98"),
                ("A woman running.", "action", "0.97"),
                ("A blue boat.", "color", "0.96"),
                ("A quiet lake.", "", "0.95"),
            ]
        )
    ]
    duplicate = dict(rows[-1], record_id="duplicate", siglip_score="1.0")
    rows.append(duplicate)
    selected = balanced_select(rows, target=5)
    assert len(selected) == 5
    assert len({row["image_path"] for row in selected}) == 5
    assert capability_bucket(rows[1]) == "spatial_relation"

    leakage = LeakageIndex(["A dog in a city park"], threshold=0.8)
    assert leakage.matches("A dog in a city park.")
    assert leakage.matches("A small dog in a city park")
    assert not leakage.matches("A dog beside a lake")


def test_grounding_cascade_finalize_excludes_qwen_rejections_and_errors(tmp_path) -> None:
    scored = tmp_path / "scored.csv"
    rows = []
    for index in range(12):
        rows.append(
            {
                "record_id": f"r{index}",
                "caption": f"A visible object number {index}",
                "image_path": f"/{index}.jpg",
                "capabilities": "scene" if index % 2 else "count",
                "siglip_status": "ok",
                "siglip_score": str(0.99 - index * 0.01),
                "siglip_logit": str(5 - index * 0.1),
            }
        )
    write_manifest(scored, rows)
    qwen_jsonl = tmp_path / "qwen.jsonl"
    qwen_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"record_id": "r0", "accepted": True}),
                json.dumps({"record_id": "r1", "accepted": False}),
                json.dumps({"record_id": "r2", "error": "parse"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    accepted = tmp_path / "accepted.csv"
    write_manifest(
        accepted,
        [{"record_id": "r0", "caption": rows[0]["caption"], "qwen36_confidence": "0.95"}],
    )
    candidate = tmp_path / "candidate.csv"
    high = tmp_path / "high.csv"
    summary = tmp_path / "summary.json"
    mixture = tmp_path / "mixture.json"
    subprocess.run(
        [
            sys.executable,
            "tools/prepare_image_bridge_grounding_cascade.py",
            "finalize",
            "--scored-manifest",
            str(scored),
            "--qwen-jsonl",
            str(qwen_jsonl),
            "--qwen-accepted-manifest",
            str(accepted),
            "--candidate-output",
            str(candidate),
            "--high-precision-output",
            str(high),
            "--summary-output",
            str(summary),
            "--mixture-output",
            str(mixture),
            "--candidate-target",
            "8",
            "--high-precision-target",
            "5",
        ],
        check=True,
    )
    candidate_ids = {row["record_id"] for row in read_manifest(candidate)}
    high_rows = read_manifest(high)
    high_ids = {row["record_id"] for row in high_rows}
    assert "r1" not in candidate_ids
    assert "r2" in candidate_ids
    assert "r2" not in high_ids
    assert "r0" in high_ids
    r0 = next(row for row in high_rows if row["record_id"] == "r0")
    assert r0["verification_source"] == "qwen36"


def test_siglip_batch_scoring_isolates_unreadable_images(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 16), "green").save(image_path)

    class FakeScorer:
        def score(self, images, captions):
            assert len(images) == len(captions) == 1
            return [0.9], [2.2]

    rows = [
        {"record_id": "ok", "caption": "A green image.", "image_path": str(image_path)},
        {
            "record_id": "bad",
            "caption": "A missing image.",
            "image_path": str(tmp_path / "missing.png"),
        },
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = score_batch(
            FakeScorer(),
            rows,
            model_id="fake-siglip",
            revision="test",
            max_image_edge=16,
            executor=executor,
        )
    assert results[0]["siglip_status"] == "ok"
    assert results[0]["siglip_logit"] == "2.20000000"
    assert results[1]["siglip_status"] == "error"
    assert "FileNotFoundError" in str(results[1]["siglip_error"])
