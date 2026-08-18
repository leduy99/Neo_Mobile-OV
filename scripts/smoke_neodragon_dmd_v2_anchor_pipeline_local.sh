#!/usr/bin/env bash
# One-GPU smoke for anchored DMD and separate monolithic bridge distillation.
# Run only inside a local H200 SLURM allocation.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
RUN_NAME="${RUN_NAME:-dmd_v2_anchor_pipeline_smoke_local}"
OUT_ROOT="${OUT_ROOT:-output/neodragon_pyramidal_dmd_reproduction/${RUN_NAME}}"
BRIDGE_OUT="${BRIDGE_OUT:-output/neo_monolithic_video_units_text_bridge/${RUN_NAME}}"
AUDIT_LATENT="${AUDIT_LATENT:-output/neodragon_pyramidal_dmd_inference_contract/dmd10k_scale_ablation_car_20260817/teacher_manual_multistep_cfg.pt}"
PROMPTS="${PROMPTS:-configs/prompts/neodragon_monolithic_bridge_smoke.txt}"

test -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" || {
  echo "Run this CUDA smoke through srun or sbatch." >&2
  exit 2
}
test -x "${PYTHON_BIN}"
test -f "${AUDIT_LATENT}"
test -f "${PROMPTS}"

mkdir -p "${OUT_ROOT}/fixture" "${BRIDGE_OUT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

AUDIT_LATENT="${AUDIT_LATENT}" OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

import torch

audit = torch.load(os.environ["AUDIT_LATENT"], map_location="cpu", weights_only=False)
latents = audit["latents"]
if tuple(latents.shape[0:3]) != (1, 16, 7):
    raise RuntimeError(f"Unexpected audit latent shape: {tuple(latents.shape)}")
root = Path(os.environ["OUT_ROOT"]) / "fixture"
torch.save({"latent": latents[0].contiguous()}, root / "sample_0000000.pt")
with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("latent_path", "condition_prompt"))
    writer.writeheader()
    writer.writerow({
        "latent_path": "sample_0000000.pt",
        "condition_prompt": "A vintage red car drives along a coastal road, cinematic, high detail",
    })
PY

"${PYTHON_BIN}" -u tools/train_neodragon_pyramidal_dmd.py \
  --manifest "${OUT_ROOT}/fixture/manifest.csv" \
  --output-dir "${OUT_ROOT}" \
  --resume none \
  --steps 1 \
  --batch-size 1 \
  --num-workers 0 \
  --no-include-first-unit \
  --external-anchor-alternative \
  --save-every 1 \
  --archive-every 999 \
  --no-save-resume \
  --no-archive-final \
  --log-every 1 \
  --dtype bf16

"${PYTHON_BIN}" -u tools/train_neodragon_text_bridge.py \
  --prompts "${PROMPTS}" \
  --output-dir "${BRIDGE_OUT}" \
  --steps 1 \
  --batch-size 1 \
  --lr 5e-5 \
  --parallel fsdp \
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
  --functional-ramp-steps 0 \
  --functional-batch-size 1 \
  --no-functional-include-first-unit \
  --functional-unit-policy cycle \
  --trainable-fp32 \
  --save-every 1 \
  --log-every 1

OUT_ROOT="${OUT_ROOT}" BRIDGE_OUT="${BRIDGE_OUT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch

dmd = torch.load(
    Path(os.environ["OUT_ROOT"]) / "neodragon_pyramidal_dmd_student_latest.pt",
    map_location="cpu",
    weights_only=False,
)
assert dmd["schedule"] == "pyramidal_1-1-1_external_anchor_video_units"
assert dmd["objective"]["native_unit_indices"] == list(range(1, 7))
assert dmd["history"][0]["unit"] == 1

bridge = torch.load(
    Path(os.environ["BRIDGE_OUT"]) / "neodragon_text_bridge_latest.pt",
    map_location="cpu",
    weights_only=False,
)
assert bridge["teacher_stack"]["name"] == "multistep"
assert "functional_dit" not in bridge
assert bridge["architecture"]["functional_include_first_unit"] is False
assert bridge["history"][0]["functional_unit"] == 1.0
print("DMD-v2 anchor alternative + monolithic bridge smoke: PASS")
PY
