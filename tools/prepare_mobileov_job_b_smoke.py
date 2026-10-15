#!/usr/bin/env python
"""Tiny mechanical inputs from existing local media; never use for quality training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoProcessor
from tools.prepare_stage2_media_smoke import build as generation_data
from tools.prepare_mobileov_job_b_qa import digest, put_image
from tools.data_prepare.download_alignment_images import atomic_json, sha256_file
from new_mobile_ov.training.stage1_alignment import JsonlRecords
from new_mobile_ov.training.stage1_alignment_data import load_sources, read_alignment_sample
from new_mobile_ov.training import job_b_qa as qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generation_data(args.source, args.output / "data")
    root = args.output / "qa"
    root.mkdir()
    sources = load_sources(args.output / "data")
    image_record = JsonlRecords(args.output / "data/validation/t2i.jsonl")[0]
    image = read_alignment_sample(image_record, sources)["target_frames"][0]
    video_record = JsonlRecords(args.output / "data/validation/t2v.jsonl")[0]
    video = read_alignment_sample(video_record, sources, num_video_frames=8)
    references = {"text": [], "image": [put_image(image, root)],
                  "video": [put_image(frame, root, seconds=seconds)
                            for frame, seconds in zip(video["target_frames"], video["frame_times"])]}
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    files = {}
    for split in ("train", "validation"):
        (root / split).mkdir()
        for modality in qa.MODALITIES:
            row = dict(sample_id=f"smoke:{split}:{modality}", modality=modality, media=references[modality],
                       group_id="mechanical_test_not_quality_data",
                       messages=[dict(role="user", content="What is two plus two?" if modality == "text" else "Is visual input provided?"),
                                 dict(role="assistant", content="Four." if modality == "text" else "Yes.")])
            qa.pack_answer(processor, row, qa.read_images(root, row))
            name = f"{split}/{modality}.jsonl"
            (root / name).write_text(json.dumps(row) + "\n")
            files[name] = dict(rows=1, sha256=sha256_file(root / name))
    processor_sha = digest(dict(vocabulary=processor.tokenizer.get_vocab(), chat_template=processor.chat_template,
                                image_processor=processor.image_processor.to_dict()))
    media_index = {reference["path"]: dict(path=reference["path"], sha256=reference["sha256"])
                   for references_for_mode in references.values() for reference in references_for_mode}
    atomic_json(root / "media_index.json", list(media_index.values()))
    atomic_json(root / "qa_summary.json", dict(format=qa.FORMAT, status="mechanically_verified_QA_release",
                                               processor_sha256=processor_sha, max_tokens=2048,
                                               media_index_sha256=sha256_file(root / "media_index.json"),
                                               files=files, integration_smoke=True,
                                               purpose="mechanical_gradient_test; shared_train_validation; NOT_quality_data"))
    qa.verify_release(root)
    print(f"Prepared integration-only media and QA: {args.output}")


if __name__ == "__main__":
    main()
