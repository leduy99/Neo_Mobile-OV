#!/usr/bin/env bash
#SBATCH --job-name=mov-imgverify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/mov-imgverify-%j.out
#SBATCH --error=logs/mov-imgverify-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${DATA_CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_image_data}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${DATA_ROOT:-data/image_bridge_v1}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${DATA_ROOT}/manifests/d2_grounded_candidates.csv}"
RUN_MODE="${RUN_MODE:-pilot}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
STOP_FILE="${TMPDIR%/}/image_bridge_qwen36_${SLURM_JOB_ID:-local}.stop"

case "${RUN_MODE}" in
  pilot)
    OUTPUT_JSONL="${OUTPUT_JSONL:-${DATA_ROOT}/annotations/qwen36_pilot.jsonl}"
    ACCEPTED_MANIFEST="${ACCEPTED_MANIFEST:-${DATA_ROOT}/manifests/d2_grounded_qwen_verified_pilot.csv}"
    MAX_SAMPLES="${MAX_SAMPLES:-20000}"
    ;;
  full)
    OUTPUT_JSONL="${OUTPUT_JSONL:-${DATA_ROOT}/annotations/qwen36_full.jsonl}"
    ACCEPTED_MANIFEST="${ACCEPTED_MANIFEST:-${DATA_ROOT}/manifests/d2_grounded_qwen_verified.csv}"
    MAX_SAMPLES="${MAX_SAMPLES:--1}"
    ;;
  *)
    echo "RUN_MODE must be pilot or full" >&2
    exit 2
    ;;
esac

test -x "${PYTHON_BIN}"
test -s "${INPUT_MANIFEST}"
mkdir -p logs "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${ACCEPTED_MANIFEST}")" "${TMPDIR}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "image-bridge-qwen36-${RUN_MODE}-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ARGS=(
  --input-manifest "${INPUT_MANIFEST}"
  --output-jsonl "${OUTPUT_JSONL}"
  --accepted-manifest "${ACCEPTED_MANIFEST}"
  --model-id "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dtype "${DTYPE:-bf16}"
  --max-new-tokens "${MAX_NEW_TOKENS:-384}"
  --min-confidence "${MIN_CONFIDENCE:-0.80}"
  --max-samples "${MAX_SAMPLES}"
  --log-every "${LOG_EVERY:-25}"
)
if [[ "${RESUME:-1}" == "1" ]]; then
  ARGS+=(--resume)
fi

"${PYTHON_BIN}" tools/verify_image_bridge_grounding_qwen36.py "${ARGS[@]}"

wc -l "${OUTPUT_JSONL}" "${ACCEPTED_MANIFEST}"
cat "${OUTPUT_JSONL}.summary.json"
