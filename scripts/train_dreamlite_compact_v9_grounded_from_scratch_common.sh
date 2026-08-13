#!/usr/bin/env bash

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
MOBILEO_ROOT="${MOBILEO_ROOT:-../Mobile-OV_Alpha/data}"
JOURNEYDB_PROMPTS="${JOURNEYDB_PROMPTS:-${MOBILEO_ROOT}/journeydb_pretrain/manifests/journeydb_pretrain_train_ready.csv}"
SHORT_PROMPTS="${SHORT_PROMPTS:-${MOBILEO_ROOT}/short_caption_pretrain/manifests/short_caption_pretrain_source.csv}"
VERIFIED_SHORT_PROMPTS="${VERIFIED_SHORT_PROMPTS:-data/dreamlite_grounded_manifests/shortcaption_verified_images.csv}"
OUT="${OUT:-output/dreamlite_compact_v9_grounded_from_scratch/${SLURM_JOB_ID:-local}}"
TARGET_STEP="${TARGET_STEP:-160000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-512x512@1024x1024:15,768x768@1024x1024:10,1024x1024@1024x1024:15,640x400@1280x800:15,1024x640@1280x800:20,400x640@800x1280:10,640x1024@800x1280:15}"
STOP_FILE="${TMPDIR%/}/dreamlite_v9_setup_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -f "${JOURNEYDB_PROMPTS}"
test -f "${SHORT_PROMPTS}"
test -s "${VERIFIED_SHORT_PROMPTS}"

# Preserve V8 image-only's caption distribution. The verified manifest has a
# zero main weight and is drawn only by the dedicated grounded loader below.
PROMPT_MANIFESTS="${JOURNEYDB_PROMPTS};${SHORT_PROMPTS};${VERIFIED_SHORT_PROMPTS}"
PROMPT_WEIGHTS="0.7142857143,0.2857142857,0.0"
PROMPT_NAMES="journeydb,shortcaption,shortcaption_verified"

mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV DreamLite V9 grounded functional distillation from scratch"
echo "PROMPT_MANIFESTS=${PROMPT_MANIFESTS}"
echo "PROMPT_WEIGHTS=${PROMPT_WEIGHTS} PROMPT_NAMES=${PROMPT_NAMES}"
echo "Grounded source=${VERIFIED_SHORT_PROMPTS} probability=${GROUNDED_BATCH_PROBABILITY:-0.15} weight=${GROUNDED_FUNCTIONAL_WEIGHT:-0.5}"
echo "OUT=${OUT} TARGET_STEP=${TARGET_STEP}"
echo "RESOLUTION_BUCKETS=${RESOLUTION_BUCKETS}"
nvidia-smi || true

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-v9-setup-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${CONFIG}" --skip-neodragon
cleanup_heartbeat
trap - EXIT INT TERM

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config "${CONFIG}" \
  --generation-prompt-manifests "${PROMPT_MANIFESTS}" \
  --generation-source-weights "${PROMPT_WEIGHTS}" \
  --generation-source-names "${PROMPT_NAMES}" \
  --grounded-source-names shortcaption_verified \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume none \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-4e-5}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-2000}" \
  --lr-final-scale "${LR_FINAL_SCALE:-0.1}" \
  --lr-decay-end-step "${LR_DECAY_END_STEP:-120000}" \
  --dtype bf16 \
  --resolution-buckets "${RESOLUTION_BUCKETS}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --semantic-prompt-probability "${SEMANTIC_PROMPT_PROBABILITY:-0.15}" \
  --representation-mode direct \
  --content-prefix-tokens 3 \
  --content-suffix-tokens 5 \
  --content-token-mse-weight "${CONTENT_TOKEN_MSE_WEIGHT:-0.5}" \
  --content-token-cos-weight "${CONTENT_TOKEN_COS_WEIGHT:-1.0}" \
  --wrapper-token-weight "${WRAPPER_TOKEN_WEIGHT:-0.1}" \
  --content-pooled-cos-weight "${CONTENT_POOLED_COS_WEIGHT:-0.25}" \
  --content-pooled-mse-weight "${CONTENT_POOLED_MSE_WEIGHT:-0.1}" \
  --content-token-mean-weight "${CONTENT_TOKEN_MEAN_WEIGHT:-0.025}" \
  --content-token-std-weight "${CONTENT_TOKEN_STD_WEIGHT:-0.025}" \
  --semantic-contrastive-weight "${SEMANTIC_CONTRASTIVE_WEIGHT:-0.1}" \
  --contrastive-temperature "${CONTRASTIVE_TEMPERATURE:-0.07}" \
  --global-contrastive \
  --projected-weight "${PROJECTED_WEIGHT:-1.0}" \
  --projected-content-scale "${PROJECTED_CONTENT_SCALE:-1.0}" \
  --representation-final-scale 1.0 \
  --functional-weight "${FUNCTIONAL_WEIGHT:-5.0}" \
  --functional-cos-weight "${FUNCTIONAL_COS_WEIGHT:-0.1}" \
  --functional-start-step "${FUNCTIONAL_START_STEP:-10001}" \
  --functional-ramp-steps "${FUNCTIONAL_RAMP_STEPS:-20000}" \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-2}" \
  --functional-call-weights "${FUNCTIONAL_CALL_WEIGHTS:-1,1,1,1}" \
  --grounded-functional-probability 0 \
  --grounded-batch-probability "${GROUNDED_BATCH_PROBABILITY:-0.15}" \
  --grounded-functional-weight "${GROUNDED_FUNCTIONAL_WEIGHT:-0.5}" \
  --transition-weight "${TRANSITION_WEIGHT:-1.0}" \
  --transition-cos-weight "${TRANSITION_COS_WEIGHT:-0.05}" \
  --student-state-probability "${STUDENT_STATE_PROBABILITY:-0.25}" \
  --student-state-start-step "${STUDENT_STATE_START_STEP:-30001}" \
  --student-state-ramp-steps "${STUDENT_STATE_RAMP_STEPS:-20000}" \
  --closed-loop-weight 0 \
  --training-version v9 \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-10000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
