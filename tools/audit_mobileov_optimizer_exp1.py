#!/usr/bin/env python
"""Check Exp-1 completion and archive metadata without loading models or Adam."""
import argparse
import json
from pathlib import Path
import re

POLICY = "epoch3_adam_resume_batch64_linear_decay_ema_v1"
PARENT_STEP = 283977


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing training artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_run(run, updates=(10000, 20000), *, require_complete=False, expected_world_size=None):
    run = Path(run).resolve()
    if not updates or any(update not in (10000, 20000) for update in updates):
        raise ValueError("Only the pinned 10K/20K additional-update archives can be evaluated")
    contract = read_json(run / "run_contract.json")
    spec = contract.get("optimizer_exp1", {})
    layout = (contract.get("world_size"), contract.get("accumulation"))
    if (contract.get("scope") != POLICY or spec.get("policy") != POLICY
            or spec.get("parent_step") != PARENT_STEP or spec.get("global_batch") != 64
            or spec.get("total_updates") != 20000 or contract.get("steps") != PARENT_STEP + 20000
            or layout not in ((16, 4), (8, 8))):
        raise ValueError("Not the expected epoch-3 Adam continuation with batch64 and 20K updates")
    if expected_world_size is not None and layout[0] != expected_world_size:
        raise ValueError(f"Expected training on {expected_world_size} GPUs, found {layout[0]}")
    completed = None
    if require_complete:
        completed = read_json(run / "training_complete.json")
        if (completed.get("format") != POLICY or completed.get("exp1_update") != 20000
                or completed.get("step") != PARENT_STEP + 20000 or completed.get("ema_updates") != 20000):
            raise ValueError("Training did not finish all 20K Exp-1 updates; inspect the training log")
    manifest = read_json(run / "evaluation_checkpoints.json")
    if manifest.get("format") != POLICY:
        raise ValueError("Checkpoint archive manifest is not an Exp-1 manifest")
    archives = []
    for update in sorted(set(updates)):
        for variant in ("raw", "ema"):
            key = f"{variant}_{update}"
            item = manifest.get("checkpoints", {}).get(key, {})
            name = f"stage2_dit_connector_{variant}_update{update:06d}.pt"
            if (item.get("path") != name or item.get("exp1_update") != update
                    or item.get("step") != PARENT_STEP + update or item.get("weight_variant") != variant
                    or item.get("ema_updates") != update
                    or not isinstance(item.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError(f"Missing or inconsistent archive entry: {key}")
            path = run / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty checkpoint: {path}")
            archives.append(dict(item, bytes=path.stat().st_size))
    return dict(training_run=str(run), training_world_size=layout[0], accumulation=layout[1],
                global_batch=64, completion=completed, archives=archives,
                verification="metadata_and_file_presence; full_weight_SHA256_and_tensors_checked_by_evaluator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--updates", nargs="+", type=int, default=[10000, 20000])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-world-size", type=int, choices=(8, 16))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run, args.updates, require_complete=args.require_complete,
                       expected_world_size=args.expected_world_size)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        target = args.output.resolve()
        if target.is_relative_to(args.run.resolve()):
            raise ValueError("Write the audit into the evaluation folder, not the training run")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
