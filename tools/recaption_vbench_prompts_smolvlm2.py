#!/usr/bin/env python
"""Create deterministic SmolVLM2 recaptions for the official VBench prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_smolvlm2_checkpoint  # noqa: E402
from new_mobile_ov.smolvlm2 import (  # noqa: E402
    SmolVLMForConditionalGeneration,
    load_smolvlm2_from_ckpt,
)


FORMAT_VERSION = "mobileov_vbench_smolvlm2_recaption_v2"
SYSTEM_INSTRUCTION = (
    "Rewrite the supplied VBench label into one literal video-generation caption. "
    "Preserve the original subject, count, color, action, spatial relation, camera cue, "
    "and style. Do not answer the prompt, explain it, write code, or add unrelated objects. "
    "Output only one caption sentence."
)
CODE_MARKERS = ("```", "def ", "import ", "class ", "print(", "return ", "#")
DEFINITION_MARKERS = (
    " is a ",
    " is an ",
    " refers to ",
    " is used to ",
    " traffic control device",
    " video-generated caption",
    " video-generating caption",
    " captures the essence",
    " central focus",
)
NON_CONTENT_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "it", "of", "on", "or", "the", "to", "with", "video", "frame", "still", "frozen",
    "time", "show", "showing", "scene", "view", "image", "cinematic", "detailed",
}


def normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_vbench_prompts(path: Path, max_prompts: int) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = normalize(row.get("prompt_en", ""))
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if max_prompts > 0 and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise RuntimeError(f"No English prompts found in {path}")
    return prompts


def prompt_for_recaption(processor: Any, prompt: str) -> str:
    """Use the same AutoProcessor chat contract as SmolVLM2 understanding."""

    user_text = f"{SYSTEM_INSTRUCTION}\n\nOriginal prompt: {prompt}"
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_new_tokens(processor: Any, generated: torch.Tensor, input_width: int) -> list[str]:
    new_tokens = generated[:, input_width:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return [normalize(value) for value in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def content_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
        if token.lower() not in NON_CONTENT_WORDS
    ]


def terms_match(source: str, candidate: str) -> bool:
    if source == candidate:
        return True
    if len(source) >= 5 and len(candidate) >= 5:
        return source[:4] == candidate[:4]
    return False


def literal_fallback(prompt: str) -> str:
    """Safe fallback: preserve the benchmark label exactly rather than hallucinate."""

    prompt = normalize(prompt).rstrip(".")
    return f"A detailed cinematic video of {prompt}."


def sanitize_recaption(prompt: str, generated: str) -> tuple[str, bool]:
    """Return a usable caption and whether SmolVLM2 output passed semantic checks."""

    candidate = normalize(generated)
    candidate = re.sub(r"^(assistant|caption|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0].strip(" \"'")
    words = candidate.split()
    if len(words) > 48:
        candidate = " ".join(words[:48]).rstrip(" ,;:")
    candidate_terms = content_terms(candidate)
    source_terms = content_terms(prompt)
    required = source_terms if len(source_terms) <= 2 else source_terms[:]
    coverage = all(any(terms_match(term, value) for value in candidate_terms) for term in required)
    invalid = (
        not candidate
        or len(candidate.split()) < 3
        or any(marker in candidate.lower() for marker in CODE_MARKERS)
        or any(marker in candidate.lower() for marker in DEFINITION_MARKERS)
        or not coverage
    )
    if invalid:
        return literal_fallback(prompt), False
    if candidate[-1] not in ".!?":
        candidate += "."
    return candidate, True


def load_existing(path: Path, prompts: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") != FORMAT_VERSION:
        print(f"Ignoring incompatible recaption cache: {path}", flush=True)
        return {}
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid recaption file format: {path}")
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = normalize(row.get("prompt", ""))
        recaption = normalize(row.get("recaption", ""))
        if prompt and recaption:
            values[prompt] = {
                "prompt": prompt,
                "recaption": recaption,
                "raw_generation": normalize(row.get("raw_generation", recaption)),
                "used_fallback": bool(row.get("used_fallback", False)),
            }
    return {prompt: values[prompt] for prompt in prompts if prompt in values}


def write_output(
    path: Path,
    *,
    source: Path,
    records: dict[str, dict[str, Any]],
    prompts: list[str],
    args,
) -> None:
    valid_records = sum(not bool(value.get("used_fallback", False)) for value in records.values())
    payload = {
        "format": FORMAT_VERSION,
        "source_vbench_info": str(source.resolve()),
        "model": str(args.tokenizer_model_id),
        "instruction": SYSTEM_INSTRUCTION,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "expected_records": len(prompts),
        "completed_records": len(records),
        "model_valid_records": valid_records,
        "fallback_records": len(records) - valid_records,
        "records": [
            # This writer is also a progress checkpoint. Do not require records
            # that have not been generated yet, otherwise the first batch cannot
            # be persisted or resumed after an interruption.
            records[prompt]
            for prompt in prompts
            if prompt in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_fraction(records: dict[str, dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(not bool(value.get("used_fallback", False)) for value in records.values()) / len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smolvlm2-checkpoint",
        default="checkpoints/smolvlm2_500m/smolvlm2_500m.pt",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLM2 recaptioning requires one allocated CUDA GPU.")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_prompts < 0:
        raise ValueError("batch-size/max-new-tokens must be positive and max-prompts non-negative")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("min-valid-fraction must be in [0, 1]")

    prompts = load_vbench_prompts(args.vbench_info, args.max_prompts)
    existing = {} if args.overwrite else load_existing(args.output, prompts)
    pending = [prompt for prompt in prompts if prompt not in existing]
    print(
        f"VBench SmolVLM2 recaption: total={len(prompts)} reused={len(existing)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        write_output(args.output, source=args.vbench_info, records=existing, prompts=prompts, args=args)
        if valid_fraction(existing) < args.min_valid_fraction:
            raise RuntimeError(
                f"Only {valid_fraction(existing):.1%} of existing recaptions passed quality checks; "
                "rerun with --overwrite after inspecting the model output."
            )
        return

    device = torch.device("cuda")
    checkpoint = ensure_smolvlm2_checkpoint(args.smolvlm2_checkpoint)
    # The feature-only wrapper deliberately has no ``generate`` method. Reuse
    # the same converted checkpoint through its causal-LM wrapper instead.
    model = load_smolvlm2_from_ckpt(
        checkpoint,
        device=device,
        model_class=SmolVLMForConditionalGeneration,
    )
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.tokenizer_model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        texts = [prompt_for_recaption(processor, prompt) for prompt in batch]
        encoded = processor(text=texts, return_tensors="pt", padding=True)
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw_recaptions = decode_new_tokens(processor, generated, model_inputs["input_ids"].shape[1])
        for prompt, raw_recaption in zip(batch, raw_recaptions):
            recaption, valid = sanitize_recaption(prompt, raw_recaption)
            existing[prompt] = {
                "prompt": prompt,
                "recaption": recaption,
                "raw_generation": raw_recaption,
                "used_fallback": not valid,
            }
        write_output(args.output, source=args.vbench_info, records=existing, prompts=prompts, args=args)
        complete = min(start + len(batch), len(pending))
        print(
            f"Recaptioned {complete}/{len(pending)} pending prompts "
            f"({complete / max(time.perf_counter() - started, 1e-6):.2f} prompt/s)",
            flush=True,
        )
        if start == 0:
            for prompt in batch:
                record = existing[prompt]
                print(
                    f"RAW: {prompt}\nMODEL: {record['raw_generation']}\n"
                    f"FINAL: {record['recaption']} fallback={record['used_fallback']}\n",
                    flush=True,
                )

    fraction = valid_fraction(existing)
    print(
        f"Saved {len(prompts)} deterministic recaptions to {args.output}; "
        f"model_valid_fraction={fraction:.1%}",
        flush=True,
    )
    if fraction < args.min_valid_fraction:
        raise RuntimeError(
            f"Recaption quality gate failed: valid_fraction={fraction:.1%} < "
            f"min_valid_fraction={args.min_valid_fraction:.1%}."
        )


if __name__ == "__main__":
    main()
