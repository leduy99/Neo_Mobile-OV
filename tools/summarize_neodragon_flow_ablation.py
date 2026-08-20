#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cells = {}
    for path in sorted(run_dir.glob("*/report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload["evaluation"]["overall"]
        cells[payload["cell"]] = {
            "flow_contract": payload["flow_contract"],
            "history_corrupt_max": payload["history_corrupt_max"],
            "teacher_weight": payload["teacher_weight"],
            "train_last_50": payload["last_50"],
            "parameter_drift": payload["parameter_drift"],
            "proper_contract_eval": overall,
        }
    contract_path = run_dir / "contract_audit" / "flow_contract_audit.json"
    summary = {
        "objective": "neodragon_flow_objective_ablation",
        "contract_audit": json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else None,
        "cells": cells,
    }
    output = run_dir / "ablation_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
