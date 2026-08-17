#!/usr/bin/env bash
# Run only under a local H200 SLURM allocation, for example:
# srun --partition=debug --gres=gpu:1 --cpus-per-task=16 --mem=110G --time=00:30:00 \
#   bash scripts/smoke_neodragon_monolithic_text_bridge_local.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
RUN_NAME="${RUN_NAME:-monolithic_bridge_smoke7_local}"
OUT="${OUT:-output/neo_monolithic_text_bridge/${RUN_NAME}}"
PROMPTS="${PROMPTS:-configs/prompts/neodragon_monolithic_bridge_smoke.txt}"

test -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" || {
  echo "Run this CUDA smoke through srun or sbatch." >&2
  exit 2
}
test -x "${PYTHON_BIN}"
test -f "${PROMPTS}"

mkdir -p "${OUT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u tools/train_neodragon_text_bridge.py \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --steps 7 \
  --batch-size 1 \
  --lr 5e-5 \
  --parallel none \
  --target-stack multistep \
  --raw-token-weight 0.25 \
  --normalized-token-weight 1.0 \
  --cos-weight 0.5 \
  --token-norm-weight 0.1 \
  --pooled-weight 0.25 \
  --pooled-cos-weight 0.2 \
  --relational-weight 0.1 \
  --functional-weight 1.0 \
  --functional-cos-weight 0.1 \
  --functional-start-step 1 \
  --functional-ramp-steps 0 \
  --functional-every 1 \
  --functional-batch-size 1 \
  --functional-include-first-unit \
  --functional-unit-policy cycle \
  --trainable-fp32 \
  --save-every 7 \
  --archive-every 7 \
  --log-every 1

OUT="${OUT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT"])
history = json.loads((root / "history.json").read_text(encoding="utf-8"))
assert {int(row["functional_unit"]) for row in history} == set(range(7))
payload_path = root / "neodragon_text_bridge_step000007.pt"
assert payload_path.is_file()
import torch
payload = torch.load(payload_path, map_location="cpu", weights_only=False)
assert payload["teacher_stack"]["name"] == "multistep"
assert payload["architecture"]["functional_include_first_unit"] is True
assert payload["architecture"]["functional_unit_policy"] == "cycle"
print("Monolithic Exp1 bridge smoke: PASS")
PY
