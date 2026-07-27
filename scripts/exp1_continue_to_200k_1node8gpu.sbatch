#!/usr/bin/env bash
#SBATCH --job-name=neo-exp1-200k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp1-200k-%j.out
#SBATCH --error=logs/neo-exp1-200k-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
REMOTE_EXP1="output/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
LOCAL_EXP1="checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  if [[ -f "${REMOTE_EXP1}" ]]; then
    INIT_CHECKPOINT="${REMOTE_EXP1}"
  else
    INIT_CHECKPOINT="${LOCAL_EXP1}"
  fi
fi

TARGET_STEP="${TARGET_STEP:-200000}"
LR="${LR:-1e-5}"
OUT="${OUT:-output/neo_exp1_continue_to200k/${SLURM_JOB_ID:-local}}"

mkdir -p logs

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing Exp1 checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

CURRENT_STEP="$(
  "${PYTHON_BIN}" - "${INIT_CHECKPOINT}" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
step = int(checkpoint.get("step", 0))
if step < 1:
    raise SystemExit(f"Checkpoint has an invalid step: {step}")
print(step)
PY
)"

if (( TARGET_STEP <= CURRENT_STEP )); then
  echo "Nothing to train: checkpoint_step=${CURRENT_STEP} target_step=${TARGET_STEP}"
  exit 0
fi

ADDITIONAL_STEPS=$((TARGET_STEP - CURRENT_STEP))
export INIT_CHECKPOINT TARGET_STEP ADDITIONAL_STEPS LR OUT

echo "Exp1 continuation target"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "CURRENT_STEP=${CURRENT_STEP}"
echo "TARGET_STEP=${TARGET_STEP}"
echo "ADDITIONAL_STEPS=${ADDITIONAL_STEPS}"
echo "LR=${LR}"
echo "OUT=${OUT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete; training was not launched."
  exit 0
fi

exec bash scripts/exp1_continue_64k_2epochs_1node8gpu.sbatch
