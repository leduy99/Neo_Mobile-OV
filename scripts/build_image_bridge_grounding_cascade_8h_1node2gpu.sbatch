#!/usr/bin/env bash
#SBATCH --job-name=mov-imgcascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=logs/mov-imgcascade-%j.out
#SBATCH --error=logs/mov-imgcascade-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:2

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

DATA_ENV="${DATA_CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_image_data}"
PYTHON_BIN="${DATA_ENV}/bin/python"
export PATH="${DATA_ENV}/bin:${PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${DATA_ROOT:-data/image_bridge_v1}"
MOBILEO_ROOT="${MOBILEO_ROOT:-../Mobile-OV_Alpha/data}"
JOURNEYDB_MANIFEST="${JOURNEYDB_MANIFEST:-${MOBILEO_ROOT}/journeydb_pretrain/manifests/journeydb_pretrain_train_ready.csv}"
SHORT_MANIFEST="${SHORT_MANIFEST:-${MOBILEO_ROOT}/short_caption_pretrain/manifests/short_caption_pretrain_source.csv}"
GROUNDED_SOURCE="${GROUNDED_SOURCE:-data/dreamlite_grounded_manifests/shortcaption_verified_images.csv}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${DATA_ROOT}/manifests/d2_grounded_candidates.csv}"
SCORED_MANIFEST="${SCORED_MANIFEST:-${DATA_ROOT}/manifests/d2_grounded_siglip2_scored.csv}"
QWEN_INPUT="${QWEN_INPUT:-${DATA_ROOT}/manifests/d2_grounded_qwen_adjudication.csv}"
QWEN_JSONL="${QWEN_JSONL:-${DATA_ROOT}/annotations/qwen36_cascade.jsonl}"
QWEN_ACCEPTED="${QWEN_ACCEPTED:-${DATA_ROOT}/manifests/d2_grounded_qwen_verified.csv}"
PILOT_JSONL="${PILOT_JSONL:-${DATA_ROOT}/annotations/qwen36_pilot.jsonl}"
PILOT_ACCEPTED="${PILOT_ACCEPTED:-${DATA_ROOT}/manifests/d2_grounded_qwen_verified_pilot.csv}"
CANDIDATE_OUTPUT="${CANDIDATE_OUTPUT:-${DATA_ROOT}/manifests/d2_grounded_candidate_100k.csv}"
HIGH_PRECISION_OUTPUT="${HIGH_PRECISION_OUTPUT:-${DATA_ROOT}/manifests/d2_grounded_high_precision_50k.csv}"
CASCADE_SUMMARY="${CASCADE_SUMMARY:-${DATA_ROOT}/stats/grounding_cascade_summary.json}"
CASCADE_MIXTURE="${CASCADE_MIXTURE:-${DATA_ROOT}/mixtures/v12_grounding_cascade_70_20_10.json}"
LEAKAGE_PROMPTS="${LEAKAGE_PROMPTS:-checkpoints/neodragon_repo/prompts/vbench_prompts.txt}"
STOP_FILE="${TMPDIR%/}/image_bridge_cascade_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -s "${GROUNDED_SOURCE}"
mkdir -p logs "${DATA_ROOT}/manifests" "${DATA_ROOT}/annotations" \
  "${DATA_ROOT}/stats" "${DATA_ROOT}/mixtures" "${TMPDIR}"

echo "ImageBridge grounding cascade job=${SLURM_JOB_ID:-local} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0,1 --interval 30 --tensor-mb 4 --work-seconds 0.05 \
  --stop-file "${STOP_FILE}" \
  --label "image-bridge-cascade-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -s "${INPUT_MANIFEST}" ]]; then
  echo "Grounded candidate manifest is missing; building ImageBridge-Data-v1 first."
  test -s "${JOURNEYDB_MANIFEST}"
  test -s "${SHORT_MANIFEST}"
  "${PYTHON_BIN}" tools/build_image_bridge_data_v1.py \
    --broad-manifests "${JOURNEYDB_MANIFEST};${SHORT_MANIFEST}" \
    --broad-source-names journeydb,shortcaption \
    --grounded-manifests "${GROUNDED_SOURCE}" \
    --grounded-source-names shortcaption_grounded \
    --output-dir "${DATA_ROOT}" \
    --max-broad-records "${MAX_BROAD_RECORDS:-3000000}" \
    --max-compositional-records "${MAX_COMPOSITIONAL_RECORDS:-200000}" \
    --max-grounded-records "${MAX_GROUNDED_RECORDS:-200000}" \
    --validation-fraction "${VALIDATION_FRACTION:-0.005}" \
    --hard-validation-per-capability "${HARD_VALIDATION_PER_CAPABILITY:-100}" \
    --progress-every "${PROGRESS_EVERY:-100000}"
fi
test -s "${INPUT_MANIFEST}"

echo "Stage 1/4: score image-caption alignment with SigLIP2."
set +e
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s "${SIGLIP_TIME_BUDGET:-3h}" \
  "${PYTHON_BIN}" tools/score_image_bridge_grounding_siglip.py \
  --input-manifest "${INPUT_MANIFEST}" \
  --output-manifest "${SCORED_MANIFEST}" \
  --model-id "${SIGLIP_MODEL_ID:-google/siglip2-so400m-patch16-384}" \
  --model-revision "${SIGLIP_MODEL_REVISION:-main}" \
  --device cuda:0 \
  --dtype "${SIGLIP_DTYPE:-bf16}" \
  --batch-size "${SIGLIP_BATCH_SIZE:-64}" \
  --loader-workers "${SIGLIP_LOADER_WORKERS:-16}" \
  --max-image-edge "${SIGLIP_MAX_IMAGE_EDGE:-1024}" \
  --max-samples "${SIGLIP_MAX_SAMPLES:-160000}" \
  --resume
SIGLIP_STATUS=$?
set -e
if [[ "${SIGLIP_STATUS}" != "0" && "${SIGLIP_STATUS}" != "124" ]]; then
  echo "SigLIP scoring failed with status=${SIGLIP_STATUS}." >&2
  exit "${SIGLIP_STATUS}"
fi
test -s "${SCORED_MANIFEST}"
SCORED_ROWS=$(( $(wc -l < "${SCORED_MANIFEST}") - 1 ))
if (( SCORED_ROWS < ${MIN_SCORED_ROWS:-110000} )); then
  echo "Only ${SCORED_ROWS} SigLIP rows are ready; require at least ${MIN_SCORED_ROWS:-110000}." >&2
  exit 1
fi
echo "SigLIP rows ready: ${SCORED_ROWS} (status=${SIGLIP_STATUS})."

echo "Stage 2/4: choose a balanced hard/ambiguous Qwen adjudication set."
DECISION_JSONL="${QWEN_JSONL}"
if [[ ! -s "${DECISION_JSONL}" && -s "${PILOT_JSONL}" ]]; then
  DECISION_JSONL="${PILOT_JSONL}"
fi
SELECT_ARGS=(
  select-qwen
  --scored-manifest "${SCORED_MANIFEST}"
  --output-manifest "${QWEN_INPUT}"
  --target "${QWEN_SELECTION_TARGET:-12000}"
)
if [[ -s "${DECISION_JSONL}" ]]; then
  SELECT_ARGS+=(--qwen-jsonl "${DECISION_JSONL}")
fi
if [[ -s "${LEAKAGE_PROMPTS}" ]]; then
  SELECT_ARGS+=(--leakage-prompts "${LEAKAGE_PROMPTS}")
fi
"${PYTHON_BIN}" tools/prepare_image_bridge_grounding_cascade.py "${SELECT_ARGS[@]}"

echo "Stage 3/4: adjudicate only selected hard records with Qwen3.6 on two GPUs."
QWEN_ARGS=(
  --input-manifest "${QWEN_INPUT}"
  --output-jsonl "${QWEN_JSONL}"
  --accepted-manifest "${QWEN_ACCEPTED}"
  --model-id "${QWEN_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
  --model-revision "${QWEN_MODEL_REVISION:-main}"
  --device-map "${QWEN_DEVICE_MAP:-balanced}"
  --dtype "${QWEN_DTYPE:-bf16}"
  --annotation-schema compact
  --max-new-tokens "${QWEN_MAX_NEW_TOKENS:-96}"
  --batch-size "${QWEN_BATCH_SIZE:-8}"
  --max-image-edge "${QWEN_MAX_IMAGE_EDGE:-768}"
  --min-confidence "${QWEN_MIN_CONFIDENCE:-0.80}"
  --max-samples "${QWEN_MAX_SAMPLES:--1}"
  --target-accepted "${QWEN_TARGET_ACCEPTED:-12000}"
  --log-every "${QWEN_LOG_EVERY:-50}"
  --flush-every "${QWEN_FLUSH_EVERY:-25}"
  --resume
  --retry-errors
)
if [[ -s "${PILOT_JSONL}" ]]; then
  QWEN_ARGS+=(--bootstrap-jsonl "${PILOT_JSONL}")
fi
if [[ -s "${PILOT_ACCEPTED}" ]]; then
  QWEN_ARGS+=(--bootstrap-accepted-manifest "${PILOT_ACCEPTED}")
fi
set +e
CUDA_VISIBLE_DEVICES=0,1 timeout --signal=TERM --kill-after=60s "${QWEN_TIME_BUDGET:-3h}" \
  "${PYTHON_BIN}" tools/verify_image_bridge_grounding_qwen36.py "${QWEN_ARGS[@]}"
QWEN_STATUS=$?
set -e
if [[ "${QWEN_STATUS}" != "0" && "${QWEN_STATUS}" != "124" ]]; then
  echo "Qwen exited with status=${QWEN_STATUS}; finalizing from flushed annotations." >&2
elif [[ "${QWEN_STATUS}" == "124" ]]; then
  echo "Qwen reached its bounded time budget; finalizing from flushed annotations."
fi
test -s "${QWEN_JSONL}"

echo "Stage 4/4: freeze the 100K candidate and 50K high-precision manifests."
FINAL_ARGS=(
  finalize
  --scored-manifest "${SCORED_MANIFEST}"
  --qwen-jsonl "${QWEN_JSONL}"
  --qwen-accepted-manifest "${QWEN_ACCEPTED}"
  --candidate-output "${CANDIDATE_OUTPUT}"
  --high-precision-output "${HIGH_PRECISION_OUTPUT}"
  --summary-output "${CASCADE_SUMMARY}"
  --mixture-output "${CASCADE_MIXTURE}"
  --candidate-target "${CANDIDATE_TARGET:-100000}"
  --high-precision-target "${HIGH_PRECISION_TARGET:-50000}"
)
if [[ -s "${LEAKAGE_PROMPTS}" ]]; then
  FINAL_ARGS+=(--leakage-prompts "${LEAKAGE_PROMPTS}")
fi
"${PYTHON_BIN}" tools/prepare_image_bridge_grounding_cascade.py "${FINAL_ARGS[@]}"

wc -l "${CANDIDATE_OUTPUT}" "${HIGH_PRECISION_OUTPUT}" "${QWEN_ACCEPTED}"
cat "${CASCADE_SUMMARY}"
if [[ "${QWEN_STATUS}" != "0" && "${QWEN_STATUS}" != "124" ]]; then
  echo "Dataset outputs were finalized, but Qwen failed with status=${QWEN_STATUS}." >&2
  exit "${QWEN_STATUS}"
fi
echo "Completed ImageBridge grounding cascade."
