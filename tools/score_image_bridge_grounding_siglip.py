#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SCORE_FIELDS = (
    "siglip_score",
    "siglip_logit",
    "siglip_status",
    "siglip_error",
    "siglip_model",
    "siglip_revision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score grounded image-caption pairs with a paired SigLIP2 model."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch16-384")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loader-workers", type=int, default=16)
    parser.add_argument("--max-image-edge", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--flush-every", type=int, default=250)
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
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("record_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("record_id", "")).strip()
        }


def load_image(path: str, max_image_edge: int) -> Image.Image:
    with Image.open(path) as image:
        loaded = image.convert("RGB").copy()
    if max_image_edge > 0 and max(loaded.size) > max_image_edge:
        loaded.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
    return loaded


class SiglipPairScorer:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
            )
        self.model = self.model.to(device).eval()
        self.device = device
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.max_text_tokens = int(
            getattr(text_config, "max_position_embeddings", 64)
        )
        self.resolved_revision = str(
            getattr(self.model.config, "_commit_hash", None) or revision
        )

    @torch.inference_mode()
    def _score_once(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
    ) -> tuple[list[float], list[float]]:
        inputs = self.processor(
            text=list(captions),
            images=list(images),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = self.model(**inputs)
        logits = getattr(outputs, "logits_per_image", None)
        if logits is None or logits.ndim != 2:
            raise RuntimeError("SigLIP output does not contain pairwise logits_per_image")
        if logits.shape[0] != len(images) or logits.shape[1] != len(captions):
            raise RuntimeError(
                f"Unexpected SigLIP logits shape {tuple(logits.shape)} for "
                f"batch={len(images)}"
            )
        paired_logits = torch.diagonal(logits.float()).cpu()
        scores = paired_logits.sigmoid()
        return scores.tolist(), paired_logits.tolist()

    def score(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
    ) -> tuple[list[float], list[float]]:
        try:
            return self._score_once(images, captions)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or len(images) <= 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(images) // 2
            left_scores, left_logits = self.score(
                images[:midpoint], captions[:midpoint]
            )
            right_scores, right_logits = self.score(
                images[midpoint:], captions[midpoint:]
            )
            return left_scores + right_scores, left_logits + right_logits


def score_batch(
    scorer: SiglipPairScorer,
    rows: Sequence[dict[str, str]],
    *,
    model_id: str,
    revision: str,
    max_image_edge: int,
    executor: ThreadPoolExecutor,
) -> list[dict[str, object]]:
    loaded = list(
        executor.map(
            lambda row: _safe_load(str(row.get("image_path", "")), max_image_edge),
            rows,
        )
    )
    valid_indices = [index for index, (image, _) in enumerate(loaded) if image is not None]
    scored: dict[int, tuple[float, float]] = {}
    if valid_indices:
        images = [loaded[index][0] for index in valid_indices]
        captions = [str(rows[index].get("caption", "")).strip() for index in valid_indices]
        scores, logits = scorer.score(images, captions)
        scored = {
            index: (score, logit)
            for index, score, logit in zip(valid_indices, scores, logits)
        }

    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        result: dict[str, object] = dict(row)
        result.update(
            {
                "siglip_model": model_id,
                "siglip_revision": revision,
                "siglip_score": "",
                "siglip_logit": "",
                "siglip_status": "error",
                "siglip_error": loaded[index][1] or "",
            }
        )
        if index in scored:
            score, logit = scored[index]
            if not math.isfinite(score) or not math.isfinite(logit):
                result["siglip_error"] = "non-finite model score"
            else:
                result.update(
                    {
                        "siglip_score": f"{score:.8f}",
                        "siglip_logit": f"{logit:.8f}",
                        "siglip_status": "ok",
                        "siglip_error": "",
                    }
                )
        output.append(result)
    return output


def _safe_load(path: str, max_image_edge: int) -> tuple[Image.Image | None, str | None]:
    try:
        if not path:
            raise FileNotFoundError("empty image_path")
        return load_image(path, max_image_edge), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.loader_workers <= 0 or args.flush_every <= 0:
        raise ValueError("batch-size, loader-workers, and flush-every must be positive")
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input manifest: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output_path}. Pass --resume to continue.")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {input_path}")
        input_fields = list(reader.fieldnames)
    output_fields = [*input_fields, *(field for field in SCORE_FIELDS if field not in input_fields)]
    processed = load_processed(output_path) if args.resume else set()
    remaining_budget = args.max_samples
    if args.max_samples > 0:
        remaining_budget = max(0, args.max_samples - len(processed))
        if remaining_budget == 0:
            print(
                f"SigLIP target already reached: {len(processed)}/{args.max_samples}",
                flush=True,
            )
            return
    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in processed:
                continue
            rows.append(row)
            if remaining_budget > 0 and len(rows) >= remaining_budget:
                break
    if not rows:
        print("No pending SigLIP candidates.", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scorer = SiglipPairScorer(
        model_id=args.model_id,
        revision=args.model_revision,
        device=device,
        dtype=torch_dtype(args.dtype),
    )
    mode = "a" if args.resume and output_path.exists() else "w"
    ok = errors = processed_this_run = 0
    with ThreadPoolExecutor(max_workers=args.loader_workers) as executor, output_path.open(
        mode, encoding="utf-8", newline=""
    ) as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for start in range(0, len(rows), args.batch_size):
            results = score_batch(
                scorer,
                rows[start : start + args.batch_size],
                model_id=args.model_id,
                revision=scorer.resolved_revision,
                max_image_edge=args.max_image_edge,
                executor=executor,
            )
            writer.writerows(results)
            ok += sum(row["siglip_status"] == "ok" for row in results)
            errors += sum(row["siglip_status"] != "ok" for row in results)
            processed_this_run += len(results)
            if processed_this_run % args.flush_every < len(results):
                output_handle.flush()
            if args.log_every > 0 and processed_this_run % args.log_every < len(results):
                print(
                    f"siglip_scored={processed_this_run}/{len(rows)} ok={ok} errors={errors}",
                    flush=True,
                )
        output_handle.flush()

    summary = {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "model_id": args.model_id,
        "requested_revision": args.model_revision,
        "resolved_revision": scorer.resolved_revision,
        "processed_before": len(processed),
        "processed_this_run": processed_this_run,
        "ok_this_run": ok,
        "errors_this_run": errors,
        "batch_size": args.batch_size,
        "loader_workers": args.loader_workers,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
        },
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
