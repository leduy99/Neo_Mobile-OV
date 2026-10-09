"""Validate and preserve the completed 1x samples when expanding to more seeds."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from vbench_joint_native_protocol import atomic_json, fingerprint, read_json, sample_items, sha256


# Both completed, GPU-validated local 1x runs used this generator. Its only
# successor change is passing the explicit seed mapping to sample_items.
REUSABLE_1X_GENERATOR = "0d085930ee3642aa9b23d1398135ae7142d2904577c7859b21653c0adf0fbedf"
COMMON_FIELDS = ("format", "full_info_sha256", "seed", "anchor", "base_video", "output", "quicksr",
                 "image_config", "video_config", "compat_code_sha256", "vbench_source_commit")


def source_contract(target, source, system):
    if source.get("samples_per_prompt") != 1 or system not in source.get("selected_systems", source.get("systems", {})):
        raise ValueError(f"Source is not a completed 1x run of {system}")
    if source.get("prompts", [])[:len(target["prompts"])] != target["prompts"]:
        raise ValueError("Cannot reuse samples with changed prompt order")
    for name in COMMON_FIELDS:
        if source.get(name) != target.get(name):
            raise ValueError(f"Cannot reuse 1x samples: changed {name}")
    if source.get("generation_code_sha256") not in (REUSABLE_1X_GENERATOR, target["generation_code_sha256"]):
        raise ValueError("Source generator version has not been validated for reuse")
    if source.get("systems", {}).get(system) != target.get("systems", {}).get(system):
        raise ValueError("Generator/sampling configuration changed")
    for key in (("joint", "image", "smolvlm2", "quicksr") if system == "joint_v2" else ("quicksr",)):
        if source["weights"][key]["sha256"] != target["weights"][key]["sha256"]:
            raise ValueError(f"Cannot reuse samples: changed {key} weights")
    for key in ("native_source_sha256", "native_source_commit", "native_model_dir"):
        if source["assets"][key] != target["assets"][key]:
            raise ValueError(f"Cannot reuse samples: changed {key}")
    files = {item["file"]: item for item in target["assets"]["native_files"]}
    for item in source["assets"]["native_files"]:
        if files.get(item["file"]) != item:
            raise ValueError(f"Changed backbone asset: {item['file']}")
    if system == "joint_v2" and source["assets"]["dreamlite_files"] != target["assets"]["dreamlite_files"]:
        raise ValueError("DreamLite generator assets changed")


def install_media(source, target, expected_sha):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != expected_sha:
            raise ValueError(f"Refusing to overwrite a conflicting sample: {target}")
        return
    try:
        # Writers replace final files atomically; linking preserves the originals.
        os.link(source, target)
    except OSError:
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected_sha:
                raise ValueError("Copied sample failed its checksum")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def reuse_samples(output, contract, specifications):
    if contract.get("seed_mapping") != "extendable" or contract["samples_per_prompt"] <= 1:
        raise ValueError("1x expansion requires the extendable seed mapping")
    targets = [item for item in sample_items(contract["prompts"], contract["samples_per_prompt"],
                                            contract["seed"], "extendable") if item["sample_index"] == 0]
    contract_sha = fingerprint(contract)
    checked, sources = [], {}
    for specification in specifications:
        system, separator, value = specification.partition("=")
        if not separator or not value or system not in contract["selected_systems"] or system in sources:
            raise ValueError(f"Invalid/duplicate reuse specification: {specification}")
        root = Path(value).resolve()
        if root == output.resolve():
            raise ValueError("Reuse source and target must be separate")
        source = read_json(root / "run_contract.json")
        receipt = read_json(root / "generation_summary.json")
        if receipt.get("status") != "complete" or receipt.get("contract_sha256") != fingerprint(source):
            raise ValueError(f"Source generation is not certified complete: {root}")
        source_contract(contract, source, system)
        old_items = list(sample_items(source["prompts"], 1, source["seed"], source.get("seed_mapping", "legacy")))
        sources[system] = {"run": str(root), "source_contract_sha256": fingerprint(source), "reused": len(targets)}
        for item in targets:
            old_item = old_items[item["prompt_index"]]
            if old_item != item:
                raise ValueError(f"Sample-0 seed changed: {system}/{item['key']}")
            folder = root / system
            record_path = folder / "records" / f"{item['key']}.json"
            record = read_json(record_path)
            anchor_record = read_json(folder / "anchor_records" / f"{item['key']}.json")
            video = folder / "videos" / item["filename"]
            anchor = folder / "anchors" / f"{item['key']}.png"
            if (record.get("sample") != old_item or record.get("system") != system
                    or record.get("contract_sha256") != fingerprint(source) or record.get("shape") != contract["output"]
                    or anchor_record.get("sample") != old_item or anchor_record.get("contract_sha256") != fingerprint(source)
                    or record.get("video_sha256") != sha256(video) or record.get("anchor_sha256") != sha256(anchor)
                    or anchor_record.get("sha256") != record["anchor_sha256"]):
                raise ValueError(f"Unverified source sample: {system}/{item['key']}")
            checked.append((system, item, video, anchor, record, record_path))
    # Validate every requested source before installing any old media.
    for system, item, video, anchor, record, record_path in checked:
        folder = output / system
        install_media(video, folder / "videos" / item["filename"], record["video_sha256"])
        install_media(anchor, folder / "anchors" / f"{item['key']}.png", record["anchor_sha256"])
        provenance = {**sources[system], "source_record_sha256": sha256(record_path),
                      "source_generation_code_sha256": read_json(Path(sources[system]["run"]) / "run_contract.json")["generation_code_sha256"]}
        atomic_json(folder / "records" / f"{item['key']}.json", {
            **record, "sample": item, "contract_sha256": contract_sha, "reused_from_1x": provenance,
        })
        atomic_json(folder / "anchor_records" / f"{item['key']}.json", {
            "sample": item, "contract_sha256": contract_sha, "sha256": record["anchor_sha256"], "reused_from_1x": provenance,
        })
    summary = {"contract_sha256": contract_sha, "sources": sources, "reused_videos": len(checked),
               "source_files_modified": False, "old_scores_reused": False,
               "note": "Sample-0 video/image bytes preserved; new metadata binds them to the expanded plan and retains source provenance."}
    atomic_json(output / "reuse_summary.json", summary)
    print(f"Preserved {len(checked)} verified sample-0 videos; only additional seeds need generation.", flush=True)
    return summary
