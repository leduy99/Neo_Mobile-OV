#!/usr/bin/env python
"""Create a fixed, balanced VBench prompt subset for comparative experiments.

The output remains a regular ``VBench_full_info.json``-compatible list, so the
official VBench evaluator can score exactly the generated subset.  The score is
therefore comparative rather than a replacement for the 944-prompt benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def normalize_prompt(value: object) -> str:
    return " ".join(str(value).strip().split())


def stable_rank(seed: int, group: tuple[str, ...], prompt: str) -> str:
    value = f"{seed}|{'|'.join(group)}|{prompt}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def prompt_dimensions(rows: list[dict]) -> dict[str, tuple[str, ...]]:
    dimensions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt = normalize_prompt(row.get("prompt_en", ""))
        if not prompt:
            continue
        for dimension in row.get("dimension", []):
            dimensions[prompt].add(str(dimension))
    if not dimensions:
        raise RuntimeError("The VBench metadata contains no usable prompt_en rows.")
    return {prompt: tuple(sorted(values)) for prompt, values in dimensions.items()}


def select_prompts(
    prompt_to_dimensions: dict[str, tuple[str, ...]],
    *,
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, list[str]]]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if count > len(prompt_to_dimensions):
        raise ValueError(
            f"Requested {count} prompts, but metadata has only {len(prompt_to_dimensions)} unique prompts."
        )

    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for prompt, dimensions in prompt_to_dimensions.items():
        groups[dimensions].append(prompt)
    for dimensions, prompts in groups.items():
        prompts.sort(key=lambda prompt: stable_rank(seed, dimensions, prompt))

    # VBench groups prompts by evaluation-dimension combinations.  Allocating one
    # quota per group preserves coverage of every benchmark dimension and avoids
    # over-sampling the large, easy prompt groups.
    ordered_groups = sorted(groups)
    base, remainder = divmod(count, len(ordered_groups))
    selected: list[str] = []
    leftovers: list[str] = []
    group_report: dict[str, list[str]] = {}
    for index, dimensions in enumerate(ordered_groups):
        quota = base + (1 if index < remainder else 0)
        candidates = groups[dimensions]
        chosen = candidates[:quota]
        selected.extend(chosen)
        leftovers.extend(candidates[quota:])
        group_report["|".join(dimensions)] = chosen

    if len(selected) < count:
        leftovers.sort(key=lambda prompt: stable_rank(seed, ("fill",), prompt))
        selected.extend(leftovers[: count - len(selected)])
    if len(selected) != count:
        raise RuntimeError(f"Internal selection error: selected={len(selected)} count={count}")
    return selected, group_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list in {args.input}, received {type(rows).__name__}.")
    prompt_to_dimensions = prompt_dimensions(rows)
    selected, group_report = select_prompts(
        prompt_to_dimensions,
        count=args.count,
        seed=args.seed,
    )
    selected_set = set(selected)
    subset = [row for row in rows if normalize_prompt(row.get("prompt_en", "")) in selected_set]
    selected_dimensions = prompt_dimensions(subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "protocol": "stratified comparative VBench subset; not the canonical full 944-prompt score",
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "requested_unique_prompts": args.count,
        "selected_unique_prompts": len(selected),
        "selected_rows": len(subset),
        "selected_prompts": selected,
        "dimension_group_counts": {name: len(prompts) for name, prompts in group_report.items()},
        "selected_dimension_coverage": {
            dimension: sum(dimension in values for values in selected_dimensions.values())
            for dimension in sorted({item for values in selected_dimensions.values() for item in values})
        },
    }
    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
