#!/usr/bin/env python
"""Create deterministic SmolVLM2 recaptions for the official VBench prompts."""

from __future__ import annotations

import argparse
import json
import os
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


SYSTEM_INSTRUCTION = (
    "Rewrite the supplied video-generation prompt as one concise, concrete visual "
    "description for a generative model. Preserve every stated subject, count, color, "
    "action, spatial relation, camera cue, and style. Do not add new objects, events, "
    "or attributes. Output only the rewritten description, with no explanation."
)


def normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_vbench_prompts(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = normalize(row.get("prompt_en", ""))
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    if not prompts:
        raise RuntimeError(f"No English prompts found in {path}")
    return prompts


def prompt_for_recaption(tokenizer: Any, prompt: str) -> str:
    user_text = f"{SYSTEM_INSTRUCTION}\n\nOriginal prompt: {prompt}"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            # Some serialized tokenizers do not retain their chat template. The
            # instruction format remains valid for the underlying instruction model.
            pass
    return f"User: {user_text}\nAssistant:"


def decode_new_tokens(tokenizer: Any, generated: torch.Tensor, input_width: int) -> list[str]:
    new_tokens = generated[:, input_width:]
    return [normalize(value) for value in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def load_existing(path: Path, prompts: list[str]) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid recaption file format: {path}")
    values: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = normalize(row.get("prompt", ""))
        recaption = normalize(row.get("recaption", ""))
        if prompt and recaption:
            values[prompt] = recaption
    return {prompt: values[prompt] for prompt in prompts if prompt in values}


def write_output(path: Path, *, source: Path, records: dict[str, str], prompts: list[str], args) -> None:
    payload = {
        "format": "mobileov_vbench_smolvlm2_recaption_v1",
        "source_vbench_info": str(source.resolve()),
        "model": str(args.tokenizer_model_id),
        "instruction": SYSTEM_INSTRUCTION,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "expected_records": len(prompts),
        "completed_records": len(records),
        "records": [
            # This writer is also a progress checkpoint. Do not require records
            # that have not been generated yet, otherwise the first batch cannot
            # be persisted or resumed after an interruption.
            {"prompt": prompt, "recaption": records[prompt]}
            for prompt in prompts
            if prompt in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLM2 recaptioning requires one allocated CUDA GPU.")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and max-new-tokens must be positive")

    prompts = load_vbench_prompts(args.vbench_info)
    existing = {} if args.overwrite else load_existing(args.output, prompts)
    pending = [prompt for prompt in prompts if prompt not in existing]
    print(
        f"VBench SmolVLM2 recaption: total={len(prompts)} reused={len(existing)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        write_output(args.output, source=args.vbench_info, records=existing, prompts=prompts, args=args)
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
    tokenizer = model.get_tokenizer()
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        texts = [prompt_for_recaption(tokenizer, prompt) for prompt in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
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
        recaptions = decode_new_tokens(tokenizer, generated, model_inputs["input_ids"].shape[1])
        for prompt, recaption in zip(batch, recaptions):
            if not recaption:
                raise RuntimeError(f"SmolVLM2 returned an empty recaption for: {prompt!r}")
            existing[prompt] = recaption
        write_output(args.output, source=args.vbench_info, records=existing, prompts=prompts, args=args)
        complete = min(start + len(batch), len(pending))
        print(
            f"Recaptioned {complete}/{len(pending)} pending prompts "
            f"({complete / max(time.perf_counter() - started, 1e-6):.2f} prompt/s)",
            flush=True,
        )

    print(f"Saved {len(prompts)} deterministic recaptions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
