#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-image-edge", type=int, default=1024)
    parser.add_argument(
        "--annotation-schema",
        choices=("compact", "detailed"),
        default="compact",
    )
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--target-accepted", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--max-consecutive-errors", type=int, default=20)
    parser.add_argument("--bootstrap-jsonl", default="")
    parser.add_argument("--bootstrap-accepted-manifest", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry IDs whose latest annotation contains an error.",
    )
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


def load_processed(path: Path, *, retry_errors: bool = False) -> set[str]:
    states: dict[str, bool] = {}
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("record_id", "")).strip()
            if record_id:
                states[record_id] = not bool(record.get("error"))
    return {
        record_id
        for record_id, succeeded in states.items()
        if succeeded or not retry_errors
    }


def load_csv_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("record_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("record_id", "")).strip()
        }


def merge_jsonl_records(source: Path, target: Path) -> int:
    """Merge unseen pilot annotations into the canonical full-run output."""
    if not source.is_file() or source.resolve() == target.resolve():
        return 0
    seen = load_processed(target)
    merged = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("a", encoding="utf-8") as dst:
        for line in src:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("record_id", "")).strip()
            if not record_id or record_id in seen:
                continue
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")
            seen.add(record_id)
            merged += 1
    return merged


def merge_csv_records(source: Path, target: Path, fieldnames: Sequence[str]) -> int:
    """Merge unseen pilot accepted rows into the canonical accepted manifest."""
    if not source.is_file() or source.resolve() == target.resolve():
        return 0
    seen = load_csv_ids(target)
    append = target.is_file() and target.stat().st_size > 0
    write_fieldnames = list(fieldnames)
    if append:
        with target.open("r", encoding="utf-8", newline="") as existing:
            existing_reader = csv.DictReader(existing)
            if existing_reader.fieldnames:
                write_fieldnames = list(existing_reader.fieldnames)
    merged = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, target.open(
        "a" if append else "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=write_fieldnames, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for row in reader:
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in seen:
                continue
            writer.writerow(row)
            seen.add(record_id)
            merged += 1
    return merged


def ensure_csv_fields(path: Path, fieldnames: Sequence[str]) -> None:
    """Upgrade a resumable accepted CSV before appending newly introduced fields."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = list(reader.fieldnames or ())
        if all(field in existing_fields for field in fieldnames):
            return
        rows = list(reader)
    upgraded_fields = list(dict.fromkeys([*existing_fields, *fieldnames]))
    temporary = path.with_suffix(path.suffix + ".schema.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=upgraded_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    failed_claims = payload.get("failed_claims", [])
    if not isinstance(failed_claims, list):
        raise ValueError("failed_claims must be a list")
    annotation["failed_claims"] = failed_claims
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


def verification_instruction(caption: str, *, schema: str = "detailed") -> str:
    if schema == "compact":
        return f"""Decide whether every central factual claim in the caption is visibly supported by the image.

Caption: {caption}

Return JSON only:
{{"caption_supported": true, "confidence": 0.0, "failed_claims": []}}

Use caption_supported=false when a central object, count, color/attribute binding, spatial relation, scene, or action conflicts with visible evidence. Put only short unsupported claims in failed_claims. Do not infer hidden facts."""
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
        max_image_edge: int,
        annotation_schema: str = "detailed",
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
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
        self.max_new_tokens = int(max_new_tokens)
        self.max_image_edge = int(max_image_edge)
        self.annotation_schema = annotation_schema
        self.resolved_revision = str(
            getattr(self.model.config, "_commit_hash", None) or model_revision
        )

    @torch.inference_mode()
    def _load_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            image = image.convert("RGB").copy()
        if self.max_image_edge > 0 and max(image.size) > self.max_image_edge:
            image.thumbnail(
                (self.max_image_edge, self.max_image_edge),
                Image.Resampling.LANCZOS,
            )
        return image

    def _generate_batch(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
    ) -> list[str]:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": verification_instruction(
                                caption,
                                schema=getattr(self, "annotation_schema", "detailed"),
                            ),
                        },
                    ],
                }
            ]
            for image, caption in zip(images, captions)
        ]
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(conversations, **kwargs)
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
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )

    def _generate_with_oom_split(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
    ) -> list[str]:
        try:
            return self._generate_batch(images, captions)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or len(images) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(images) // 2
            print(
                f"Batch OOM at size={len(images)}; retrying as "
                f"{midpoint}+{len(images) - midpoint}.",
                flush=True,
            )
            return self._generate_with_oom_split(
                images[:midpoint], captions[:midpoint]
            ) + self._generate_with_oom_split(images[midpoint:], captions[midpoint:])

    @torch.inference_mode()
    def verify_batch(
        self,
        items: Sequence[tuple[str, str]],
    ) -> list[tuple[dict[str, Any] | None, str, str | None]]:
        results: list[tuple[dict[str, Any] | None, str, str | None] | None] = [
            None
        ] * len(items)
        valid_indices: list[int] = []
        images: list[Image.Image] = []
        captions: list[str] = []
        for index, (image_path, caption) in enumerate(items):
            try:
                images.append(self._load_image(image_path))
                captions.append(caption)
                valid_indices.append(index)
            except Exception as exc:
                results[index] = (None, "", f"{type(exc).__name__}: {exc}")

        if images:
            responses = self._generate_with_oom_split(images, captions)
            if len(responses) != len(valid_indices):
                raise RuntimeError(
                    f"Verifier returned {len(responses)} responses for "
                    f"{len(valid_indices)} inputs"
                )
            for index, response in zip(valid_indices, responses):
                try:
                    results[index] = (
                        validate_annotation(extract_json(response)),
                        response,
                        None,
                    )
                except Exception as exc:
                    results[index] = (
                        None,
                        response,
                        f"{type(exc).__name__}: {exc}",
                    )

        return [
            result
            if result is not None
            else (None, "", "RuntimeError: missing verifier result")
            for result in results
        ]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if args.batch_size <= 0 or args.flush_every <= 0:
        raise ValueError("batch_size and flush_every must be positive")
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

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {input_path}")
        fieldnames = list(reader.fieldnames)

    accepted_fields = [
        *fieldnames,
        "qwen36_confidence",
        "qwen36_objects",
        "qwen36_counts",
        "qwen36_attributes",
        "qwen36_relations",
        "qwen36_scene",
        "qwen36_action",
        "qwen36_failed_claims",
    ]
    merged_jsonl = merged_accepted = 0
    if args.resume and args.bootstrap_jsonl:
        merged_jsonl = merge_jsonl_records(Path(args.bootstrap_jsonl), output_path)
    if args.resume and args.bootstrap_accepted_manifest:
        merged_accepted = merge_csv_records(
            Path(args.bootstrap_accepted_manifest), accepted_path, accepted_fields
        )
    if merged_jsonl or merged_accepted:
        print(
            f"Bootstrapped pilot records: annotations={merged_jsonl} "
            f"accepted={merged_accepted}",
            flush=True,
        )
    if args.resume:
        ensure_csv_fields(accepted_path, accepted_fields)

    processed = (
        load_processed(output_path, retry_errors=args.retry_errors)
        if args.resume
        else set()
    )
    accepted_ids = load_csv_ids(accepted_path) if args.resume else set()
    if args.target_accepted > 0 and len(accepted_ids) >= args.target_accepted:
        print(
            f"Accepted target already reached: {len(accepted_ids)}/{args.target_accepted}",
            flush=True,
        )
        return
    remaining_sample_budget = -1
    if args.max_samples > 0:
        remaining_sample_budget = max(0, args.max_samples - len(processed))
        if remaining_sample_budget == 0:
            print(
                f"Processed target already reached: {len(processed)}/{args.max_samples}",
                flush=True,
            )
            return

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row_index, row in enumerate(reader):
            if row_index % args.num_shards != args.shard_index:
                continue
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in processed:
                continue
            rows.append(row)
            if remaining_sample_budget > 0 and len(rows) >= remaining_sample_budget:
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
        max_image_edge=args.max_image_edge,
        annotation_schema=args.annotation_schema,
    )
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
        processed_this_run = 0
        for batch_start in range(0, len(rows), args.batch_size):
            if args.target_accepted > 0 and len(accepted_ids) >= args.target_accepted:
                break
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch_results = verifier.verify_batch(
                [
                    (
                        str(row.get("image_path", "")).strip(),
                        str(row.get("caption", "")).strip(),
                    )
                    for row in batch_rows
                ]
            )
            for row, (annotation, raw_response, error) in zip(batch_rows, batch_results):
                record_id = str(row["record_id"])
                caption = str(row.get("caption", "")).strip()
                image_path = str(row.get("image_path", "")).strip()
                result: dict[str, Any] = {
                    "record_id": record_id,
                    "caption": caption,
                    "image_path": image_path,
                    "accepted": False,
                }
                if error is None and annotation is not None:
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
                                    "qwen36_failed_claims": json.dumps(
                                        annotation["failed_claims"]
                                    ),
                                }
                            )
                            writer.writerow(accepted_row)
                            accepted_ids.add(record_id)
                        accepted += 1
                    else:
                        rejected += 1
                    consecutive_errors = 0
                else:
                    result["raw_response"] = raw_response
                    result["error"] = error or "RuntimeError: missing annotation"
                    errors += 1
                    consecutive_errors += 1
                result_handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                processed_this_run += 1
                if processed_this_run % args.flush_every == 0:
                    result_handle.flush()
                    accepted_handle.flush()
                if args.log_every > 0 and processed_this_run % args.log_every == 0:
                    print(
                        f"verified={processed_this_run}/{len(rows)} "
                        f"accepted_run={accepted} accepted_total={len(accepted_ids)} "
                        f"rejected={rejected} errors={errors}",
                        flush=True,
                    )
                if consecutive_errors >= args.max_consecutive_errors:
                    raise RuntimeError(
                        f"Stopped after {consecutive_errors} consecutive verification errors."
                    )
            result_handle.flush()
            accepted_handle.flush()
    summary = {
        "input_manifest": str(input_path),
        "output_jsonl": str(output_path),
        "accepted_manifest": str(accepted_path),
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": verifier.resolved_revision,
        "min_confidence": args.min_confidence,
        "processed_this_run": processed_this_run,
        "processed_total": len(processed) + processed_this_run,
        "accepted_this_run": accepted,
        "accepted_total": len(accepted_ids),
        "rejected_this_run": rejected,
        "errors_this_run": errors,
        "acceptance_rate": accepted / max(accepted + rejected, 1),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "max_image_edge": args.max_image_edge,
        "annotation_schema": args.annotation_schema,
        "retry_errors": args.retry_errors,
        "target_accepted": args.target_accepted,
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
