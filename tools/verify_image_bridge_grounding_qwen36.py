#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_LIST_FIELDS = ("objects", "counts", "attributes", "relations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visually verify ImageBridge-Data-v1 grounded candidates with Qwen3.6."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--accepted-manifest", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-consecutive-errors", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_processed(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.is_file():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("record_id", "")).strip()
            if record_id:
                processed.add(record_id)
    return processed


def load_csv_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("record_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("record_id", "")).strip()
        }


def extract_json(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload = None
        for start, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            raise ValueError("Qwen response contains no valid JSON object")
    if not isinstance(payload, dict):
        raise ValueError("Qwen response is not a JSON object")
    return payload


def validate_annotation(payload: Mapping[str, Any]) -> dict[str, Any]:
    supported = payload.get("caption_supported")
    if not isinstance(supported, bool):
        raise ValueError("caption_supported must be boolean")
    confidence = float(payload.get("confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    annotation: dict[str, Any] = {
        "caption_supported": supported,
        "confidence": confidence,
    }
    for field in REQUIRED_LIST_FIELDS:
        value = payload.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        annotation[field] = value
    for field in ("scene", "action", "reason"):
        value = payload.get(field, "")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        annotation[field] = value or ""
    return annotation


def verification_instruction(caption: str) -> str:
    return f"""Verify whether the caption is factually supported by the image.

Caption: {caption}

Return one JSON object with exactly these fields:
{{
  "caption_supported": true,
  "confidence": 0.0,
  "objects": ["visible object"],
  "counts": [{{"object": "object", "count": 1}}],
  "attributes": [{{"object": "object", "attribute": "visible attribute"}}],
  "relations": [{{"subject": "object", "relation": "left of", "object": "object"}}],
  "scene": "visible scene or empty string",
  "action": "visible action or empty string",
  "reason": "short factual reason"
}}

Rules:
- Inspect only visible evidence. Do not infer hidden objects, identities, locations, or events.
- Mark caption_supported=false if any central object, count, attribute, relation, scene, or action conflicts with the image.
- Omit uncertain facts from the lists instead of guessing.
- Preserve exact numeric counts only when clearly visible.
- Return JSON only, without markdown or commentary."""


class QwenVisualVerifier:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        dtype: torch.dtype,
        device_map: str,
        max_new_tokens: int,
    ) -> None:
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.6 requires a recent Transformers build containing "
                "AutoModelForMultimodalLM. Upgrade the dedicated data environment; "
                "do not modify the Neo-MobileOV training environment in place."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=model_revision,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.max_new_tokens = int(max_new_tokens)
        self.resolved_revision = str(
            getattr(self.model.config, "_commit_hash", None) or model_revision
        )

    @torch.inference_mode()
    def verify(self, image_path: str, caption: str) -> tuple[dict[str, Any], str]:
        with Image.open(image_path) as image:
            image = image.convert("RGB").copy()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": verification_instruction(caption)},
                ],
            }
        ]
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(messages, **kwargs)
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[-1])
        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        generated = output[:, input_length:]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        return validate_annotation(extract_json(response)), response


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_jsonl)
    accepted_path = Path(args.accepted_manifest)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input manifest: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output_path.exists() or accepted_path.exists()):
        raise FileExistsError(
            "Output exists. Pass --resume to continue without repeating completed IDs."
        )
    processed = load_processed(output_path) if args.resume else set()
    accepted_ids = load_csv_ids(accepted_path) if args.resume else set()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {input_path}")
        fieldnames = list(reader.fieldnames)
        rows = []
        for row_index, row in enumerate(reader):
            if row_index % args.num_shards != args.shard_index:
                continue
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in processed:
                continue
            rows.append(row)
            if args.max_samples > 0 and len(rows) >= args.max_samples:
                break
    if not rows:
        print("No pending grounded candidates.", flush=True)
        return

    verifier = QwenVisualVerifier(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    accepted_fields = [
        *fieldnames,
        "qwen36_confidence",
        "qwen36_objects",
        "qwen36_counts",
        "qwen36_attributes",
        "qwen36_relations",
        "qwen36_scene",
        "qwen36_action",
    ]
    accepted_mode = "a" if args.resume and accepted_path.exists() else "w"
    output_mode = "a" if args.resume and output_path.exists() else "w"
    accepted = 0
    rejected = 0
    errors = 0
    consecutive_errors = 0
    with output_path.open(output_mode, encoding="utf-8") as result_handle, accepted_path.open(
        accepted_mode, encoding="utf-8", newline=""
    ) as accepted_handle:
        writer = csv.DictWriter(accepted_handle, fieldnames=accepted_fields)
        if accepted_mode == "w":
            writer.writeheader()
        for index, row in enumerate(rows, start=1):
            record_id = str(row["record_id"])
            caption = str(row.get("caption", "")).strip()
            image_path = str(row.get("image_path", "")).strip()
            result: dict[str, Any] = {
                "record_id": record_id,
                "caption": caption,
                "image_path": image_path,
                "accepted": False,
            }
            try:
                annotation, raw_response = verifier.verify(image_path, caption)
                is_accepted = bool(
                    annotation["caption_supported"]
                    and annotation["confidence"] >= args.min_confidence
                )
                result.update(
                    {
                        "accepted": is_accepted,
                        "annotation": annotation,
                        "raw_response": raw_response,
                    }
                )
                if is_accepted:
                    if record_id not in accepted_ids:
                        accepted_row = dict(row)
                        accepted_row.update(
                            {
                                "grounding_status": "qwen36_verified",
                                "qwen36_confidence": annotation["confidence"],
                                "qwen36_objects": json.dumps(annotation["objects"]),
                                "qwen36_counts": json.dumps(annotation["counts"]),
                                "qwen36_attributes": json.dumps(annotation["attributes"]),
                                "qwen36_relations": json.dumps(annotation["relations"]),
                                "qwen36_scene": annotation["scene"],
                                "qwen36_action": annotation["action"],
                            }
                        )
                        writer.writerow(accepted_row)
                        accepted_ids.add(record_id)
                    accepted += 1
                else:
                    rejected += 1
                consecutive_errors = 0
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
                consecutive_errors += 1
            result_handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            result_handle.flush()
            accepted_handle.flush()
            if args.log_every > 0 and index % args.log_every == 0:
                print(
                    f"verified={index}/{len(rows)} accepted={accepted} "
                    f"rejected={rejected} errors={errors}",
                    flush=True,
                )
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Stopped after {consecutive_errors} consecutive verification errors."
                )
    summary = {
        "input_manifest": str(input_path),
        "output_jsonl": str(output_path),
        "accepted_manifest": str(accepted_path),
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": verifier.resolved_revision,
        "min_confidence": args.min_confidence,
        "processed_this_run": len(rows),
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "errors_this_run": errors,
        "acceptance_rate": accepted / max(accepted + rejected, 1),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
