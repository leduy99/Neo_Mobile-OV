#!/usr/bin/env bash
# One-GPU H200 smoke for the all-seven-unit Pyramidal-DMD correction.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
RUN_NAME="${RUN_NAME:-dmd_v2_all7_smoke21_local}"
OUT_ROOT="${OUT_ROOT:-output/neodragon_pyramidal_dmd_reproduction/${RUN_NAME}}"
AUDIT_LATENT="${AUDIT_LATENT:-output/neodragon_pyramidal_dmd_inference_contract/dmd10k_scale_ablation_car_20260817/teacher_manual_multistep_cfg.pt}"

test -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" || {
  echo "Run this CUDA smoke through srun or sbatch." >&2
  exit 2
}
test -x "${PYTHON_BIN}"
test -f "${AUDIT_LATENT}"

mkdir -p "${OUT_ROOT}/fixture"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The fixture is a multi-step native T2V rollout created by the local audit.
# It lets the smoke cover every unit/stage without copying Berzelius data.
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
prompt = "A vintage red car drives along a coastal road at golden hour, the camera tracking beside it."
with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("latent_path", "condition_prompt"))
    writer.writeheader()
    writer.writerow({
        "latent_path": "sample_0000000.pt",
        "condition_prompt": prompt + ", cinematic, realistic textures, high detail, natural colours",
    })
PY

"${PYTHON_BIN}" -u tools/train_neodragon_pyramidal_dmd.py \
  --manifest "${OUT_ROOT}/fixture/manifest.csv" \
  --output-dir "${OUT_ROOT}" \
  --resume none \
  --steps 21 \
  --batch-size 1 \
  --num-workers 0 \
  --include-first-unit \
  --teacher-first-guidance 7.0 \
  --teacher-video-guidance 5.0 \
  --save-every 21 \
  --archive-every 21 \
  --log-every 1 \
  --dtype bf16

OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch

root = Path(os.environ["OUT_ROOT"])
payload = torch.load(root / "neodragon_pyramidal_dmd_student_step000021.pt", map_location="cpu", weights_only=False)
assert payload["schedule"] == "pyramidal_1-1-1_all_native_units"
assert payload["objective"]["name"] == "pyramidal_dmd_reproduction_v2_all_native_units"
assert payload["objective"]["native_unit_indices"] == list(range(7))
history = payload["history"]
assert {(row["unit"], row["stage"]) for row in history} == {(unit, stage) for unit in range(7) for stage in range(3)}
# One 21-position pass must use a single fixed student probe value.  The next
# pass advances to the next value in the four-probe cycle.
assert {row["tau"] for row in history} == {0.125}
print("DMD-v2 all-seven-unit smoke: PASS")
PY
