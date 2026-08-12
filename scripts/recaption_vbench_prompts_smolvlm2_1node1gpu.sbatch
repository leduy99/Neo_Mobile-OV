#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench-recap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mov-vbench-recap-%j.out
#SBATCH --error=logs/mov-vbench-recap-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
VBENCH_REPO="${VBENCH_REPO:-$(pwd -P)/checkpoints/vbench_repo}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
if [[ -n "${OUT:-}" ]]; then
  OUT="${OUT}"
elif [[ "${MAX_PROMPTS}" -gt 0 ]]; then
  OUT="output/vbench_smolvlm2_recaptions/vbench_smoke${MAX_PROMPTS}_smolvlm2_recap.json"
else
  OUT="output/vbench_smolvlm2_recaptions/vbench_full_smolvlm2_recap.json"
fi
STOP_FILE="${TMPDIR%/}/mov_vbench_recap_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
if [[ ! -f "${VBENCH_INFO}" ]]; then
  VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
fi
test -f "${VBENCH_INFO}"

mkdir -p logs "$(dirname "${OUT}")"
rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-vbench-recap-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "VBENCH_INFO=${VBENCH_INFO}"
echo "OUT=${OUT}"
echo "MAX_PROMPTS=${MAX_PROMPTS}"
nvidia-smi || true

"${PYTHON_BIN}" tools/recaption_vbench_prompts_smolvlm2.py \
  --vbench-info "${VBENCH_INFO}" \
  --output "${OUT}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-96}" \
  --max-prompts "${MAX_PROMPTS}" \
  --min-valid-fraction "${MIN_VALID_FRACTION:-0.90}" \
  ${EXTRA_ARGS:-}

"${PYTHON_BIN}" - "${OUT}" <<'PY'
import json
import sys

path = sys.argv[1]
records = json.load(open(path, encoding="utf-8"))["records"]
print(f"Completed VBench recaption file: {path} records={len(records)}")
for row in records[:5]:
    print(f"RAW: {row['prompt']}\nRECAPTION: {row['recaption']}\n")
PY
