#!/usr/bin/env bash
#SBATCH --job-name=mov-imgv12-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mov-imgv12-data-%j.out
#SBATCH --error=logs/mov-imgv12-data-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

# Machine-check the frozen V12 manifests and GPU-rescore a deterministic subset.
set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

DATA_ENV="${DATA_CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_image_data}"
PYTHON_BIN="${DATA_ENV}/bin/python"
DATA_ROOT="${DATA_ROOT:-data/image_bridge_v1}"
HIGH_PRECISION="${HIGH_PRECISION:-${DATA_ROOT}/manifests/d2_grounded_high_precision_50k.csv}"
QWEN_JSONL="${QWEN_JSONL:-${DATA_ROOT}/annotations/qwen36_cascade.jsonl}"
AUDIT_CSV="${AUDIT_CSV:-${DATA_ROOT}/annotations/qwen36_cascade_human_audit.csv}"
RESCORE_SUBSET="${RESCORE_SUBSET:-${DATA_ROOT}/stats/v12_siglip2_rescore_subset.csv}"
RESCORE_OUTPUT="${RESCORE_OUTPUT:-${DATA_ROOT}/stats/v12_siglip2_rescore_results.csv}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-${DATA_ROOT}/stats/v12_data_preflight.json}"
LEAKAGE_PROMPTS="${LEAKAGE_PROMPTS:-checkpoints/neodragon_repo/prompts/vbench_prompts.txt}"
RESCORE_SAMPLES="${RESCORE_SAMPLES:-2048}"
STOP_FILE="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}/image_bridge_v12_data_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -s "${HIGH_PRECISION}"
test -s "${QWEN_JSONL}"
test -s "${DATA_ROOT}/mixtures/v12_grounding_cascade_70_20_10.json"
mkdir -p logs "${DATA_ROOT}/stats" "${DATA_ROOT}/annotations" "$(dirname "${STOP_FILE}")"

export PATH="${DATA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "ImageBridge V12 data preflight job=${SLURM_JOB_ID:-local} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 30 --tensor-mb 4 --work-seconds 0.05 \
  --stop-file "${STOP_FILE}" \
  --label "image-bridge-v12-data-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -s "${AUDIT_CSV}" ]]; then
  "${PYTHON_BIN}" tools/sample_image_bridge_qwen_audit.py \
    --input-jsonl "${QWEN_JSONL}" \
    --output-csv "${AUDIT_CSV}" \
    --samples-per-class "${AUDIT_SAMPLES_PER_CLASS:-100}"
fi

if [[ "${SCORE_HUMAN_AUDIT:-0}" == "1" ]]; then
  "${PYTHON_BIN}" tools/sample_image_bridge_qwen_audit.py \
    --input-jsonl "${QWEN_JSONL}" \
    --output-csv "${AUDIT_CSV}" \
    --samples-per-class "${AUDIT_SAMPLES_PER_CLASS:-100}" \
    --minimum-agreement "${MINIMUM_HUMAN_AGREEMENT:-0.95}" \
    --score-filled-sheet
fi

head -n "$((RESCORE_SAMPLES + 1))" "${HIGH_PRECISION}" > "${RESCORE_SUBSET}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/score_image_bridge_grounding_siglip.py \
  --input-manifest "${RESCORE_SUBSET}" \
  --output-manifest "${RESCORE_OUTPUT}" \
  --model-id "${SIGLIP_MODEL_ID:-google/siglip2-so400m-patch16-384}" \
  --model-revision "${SIGLIP_MODEL_REVISION:-main}" \
  --device cuda:0 \
  --dtype "${SIGLIP_DTYPE:-bf16}" \
  --batch-size "${SIGLIP_BATCH_SIZE:-64}" \
  --loader-workers "${SIGLIP_LOADER_WORKERS:-16}" \
  --max-image-edge "${SIGLIP_MAX_IMAGE_EDGE:-1024}" \
  --resume

VALIDATE_ARGS=(
  --data-root "${DATA_ROOT}"
  --output "${PREFLIGHT_OUTPUT#${DATA_ROOT}/}"
  --rescore-manifest "${RESCORE_OUTPUT}"
  --audit-sheet "${AUDIT_CSV}"
  --decode-samples "${DECODE_SAMPLES:-2048}"
  --decode-workers "${DECODE_WORKERS:-16}"
  --expected-grounded-rows "${EXPECTED_GROUNDED_ROWS:-50000}"
)
if [[ -s "${LEAKAGE_PROMPTS}" ]]; then
  VALIDATE_ARGS+=(--leakage-prompts "${LEAKAGE_PROMPTS}")
fi
if [[ "${ALLOW_LEAKAGE:-0}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-leakage)
fi
"${PYTHON_BIN}" tools/validate_image_bridge_v12_data.py "${VALIDATE_ARGS[@]}"

wc -l \
  "${DATA_ROOT}/manifests/d1_broad_train.csv" \
  "${DATA_ROOT}/manifests/d2_compositional_train.csv" \
  "${HIGH_PRECISION}" \
  "${AUDIT_CSV}" \
  "${RESCORE_OUTPUT}"
cat "${PREFLIGHT_OUTPUT}"
echo "Machine preflight passed. Human audit sheet: ${AUDIT_CSV}"
