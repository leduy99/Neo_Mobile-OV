#!/usr/bin/env bash
#SBATCH --job-name=mov-ssd-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-ssd-v2-%j.out
#SBATCH --error=logs/mov-ssd-v2-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_ssd1b_image_bridge_v2.yaml}"
PROMPTS="${PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
HARD_PROMPTS="${HARD_PROMPTS:-configs/prompts/ssd1b_image_bridge_hard_prompts.txt}"
OUT="${OUT:-output/ssd1b_image_bridge_v2}"
TARGET_STEP="${TARGET_STEP:-100000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RESUME="${RESUME:-auto}"
HEARTBEAT_STOP_FILE="${TMPDIR%/}/ssd1b_bridge_v2_setup_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs "${OUT}"

if [[ ! -x "${TORCHRUN_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${CONDA_ENV}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS}" || ! -f "${HARD_PROMPTS}" ]]; then
  echo "Missing prompts: PROMPTS=${PROMPTS} HARD_PROMPTS=${HARD_PROMPTS}" >&2
  exit 1
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV SSD1B Image Bridge v2 distillation"
echo "CONFIG=${CONFIG} PROMPTS=${PROMPTS} HARD_PROMPTS=${HARD_PROMPTS}"
echo "OUT=${OUT} TARGET_STEP=${TARGET_STEP} NPROC_PER_NODE=${NPROC_PER_NODE} RESUME=${RESUME}"
nvidia-smi || true

rm -f "${HEARTBEAT_STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices \
  --interval 5 \
  --tensor-mb 4 \
  --work-seconds 1 \
  --stop-file "${HEARTBEAT_STOP_FILE}" \
  --label "ssd1b-bridge-v2-setup-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup_heartbeat() {
  touch "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
  if kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
    wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

sleep 3
if ! kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  echo "GPU heartbeat failed during checkpoint preparation." >&2
  exit 1
fi
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${CONFIG}"
cleanup_heartbeat
trap - EXIT INT TERM

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_ssd1b_image_bridge.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --hard-prompts "${HARD_PROMPTS}" \
  --hard-prompt-prob "${HARD_PROMPT_PROB:-0.20}" \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume "${RESUME}" \
  --objective-version v2 \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-4e-5}" \
  --dtype bf16 \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --prompt-modifier-prob "${PROMPT_MODIFIER_PROB:-0.50}" \
  --clip-l-weight 0.35 \
  --clip-big-g-weight 0.35 \
  --pooled-weight 0.50 \
  --geometry-weight 0.25 \
  --norm-weight 0.05 \
  --eos-weight 0.50 \
  --padding-weight 0.15 \
  --retrieval-weight 0.10 \
  --variance-weight 0.05 \
  --representation-final-scale 0.35 \
  --representation-decay-steps 15000 \
  --functional-weight 20.0 \
  --functional-cos-weight 0.5 \
  --functional-start-step 10001 \
  --functional-ramp-steps 5000 \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-1}" \
  --closed-loop-weight 1.0 \
  --closed-loop-prediction-weight 0.5 \
  --closed-loop-cos-weight 0.1 \
  --closed-loop-transition-weight 1.0 \
  --closed-loop-terminal-weight 1.0 \
  --closed-loop-start-step 25001 \
  --closed-loop-ramp-steps 5000 \
  --closed-loop-every "${CLOSED_LOOP_EVERY:-4}" \
  --closed-loop-batch-size "${CLOSED_LOOP_BATCH_SIZE:-1}" \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-10000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
