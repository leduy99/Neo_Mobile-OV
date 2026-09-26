#!/usr/bin/env python
"""Create a tiny local integration-test manifest; reuse media without copying it."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_mobile_ov.training.stage1_alignment import TASKS, JsonlRecords, verify_release, prepare_sample
from new_mobile_ov.training.stage2_media_epochs import first_frame_image
from new_mobile_ov.training.stage1_alignment_data import load_sources
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Smoke destination must be new; never overwrite a dataset")
    summary = verify_release(source)
    roots = load_sources(source)
    sources = json.loads((source / "sources.json").read_text())
    for key, root in roots.items():
        sources[key]["root"] = str(root)
    output.mkdir(parents=True)
    atomic_json(output / "sources.json", sources)
    counts, manifests = {}, {}
    for split in ("train", "validation"):
        (output / split).mkdir()
        for task in TASKS:
            records = JsonlRecords(source / split / f"{task}.jsonl")
            count = (5 if task == "t2v" else 3) if split == "train" else 1
            if len(records) < count:
                raise ValueError("Smoke source is too small")
            path = output / split / f"{task}.jsonl"
            with path.open("w") as stream:
                for index in range(count):
                    stream.write(json.dumps(records[index]) + "\n")
            counts[f"{split}.{task}"] = count
            manifests[f"{split}.{task}"] = dict(path=str(path.relative_to(output)), sha256=sha256_file(path))
    summary.update(counts=counts, manifests=manifests, sources=sources,
                   sources_sha256=sha256_file(output / "sources.json"), purpose="local_integration_smoke_only")
    atomic_json(output / "stage1_summary.json", summary)
    atomic_json(output / ".stage1_data_complete", summary)
    verify_release(output)
    record = JsonlRecords(output / "validation" / "t2v.jsonl")[0]
    sample = prepare_sample(record, roots)
    first_frame_image(sample["video"]).save(output / "heldout_first_frame.png")
    (output / "heldout_prompt.txt").write_text(sample["prompt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
