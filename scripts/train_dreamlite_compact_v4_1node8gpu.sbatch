#!/usr/bin/env bash
#SBATCH --job-name=mov-dream-v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-v4-%j.out
#SBATCH --error=logs/mov-dream-v4-%j.out
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
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite_compact_v4.yaml}"
GENERATION_PROMPTS="${GENERATION_PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
OUT="${OUT:-output/dreamlite_compact_v4/${SLURM_JOB_ID:-local}}"
TARGET_STEP="${TARGET_STEP:-100000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-512x512@1024x1024:20,768x768@1024x1024:10,1024x1024@1024x1024:5,640x400@1280x800:25,1024x640@1280x800:20,400x640@800x1280:12,640x1024@800x1280:8}"
STOP_FILE="${TMPDIR%/}/dreamlite_v4_setup_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs "${OUT}"
test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -f "${GENERATION_PROMPTS}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV DreamLite compact V4 distillation"
echo "CONFIG=${CONFIG} GENERATION_PROMPTS=${GENERATION_PROMPTS}"
echo "OUT=${OUT} TARGET_STEP=${TARGET_STEP} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "RESOLUTION_BUCKETS=${RESOLUTION_BUCKETS}"
nvidia-smi || true

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-v4-setup-${SLURM_JOB_ID:-local}" &
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
  --generation-prompts "${GENERATION_PROMPTS}" \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume "${RESUME:-auto}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-4e-5}" \
  --dtype bf16 \
  --resolution-buckets "${RESOLUTION_BUCKETS}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --representation-mode direct \
  --token-mse-weight "${TOKEN_MSE_WEIGHT:-0.5}" \
  --token-cos-weight "${TOKEN_COS_WEIGHT:-1.0}" \
  --token-norm-weight "${TOKEN_NORM_WEIGHT:-0.0}" \
  --pooled-mse-weight "${POOLED_MSE_WEIGHT:-0.0}" \
  --pooled-cos-weight "${POOLED_COS_WEIGHT:-0.25}" \
  --geometry-weight "${GEOMETRY_WEIGHT:-0.0}" \
  --variance-weight "${VARIANCE_WEIGHT:-0.0}" \
  --token-mean-weight "${TOKEN_MEAN_WEIGHT:-0.0}" \
  --token-std-weight "${TOKEN_STD_WEIGHT:-0.0}" \
  --projected-weight "${PROJECTED_WEIGHT:-1.0}" \
  --projected-pooled-cos-weight "${PROJECTED_POOLED_COS_WEIGHT:-0.0}" \
  --projected-moment-weight "${PROJECTED_MOMENT_WEIGHT:-0.0}" \
  --representation-final-scale 1.0 \
  --functional-weight "${FUNCTIONAL_WEIGHT:-5.0}" \
  --functional-cos-weight "${FUNCTIONAL_COS_WEIGHT:-0.1}" \
  --functional-start-step "${FUNCTIONAL_START_STEP:-5001}" \
  --functional-ramp-steps "${FUNCTIONAL_RAMP_STEPS:-10000}" \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-1}" \
  --closed-loop-weight 0 \
  --training-version v4 \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-10000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
