#!/usr/bin/env bash
#SBATCH --job-name=mov-dream-bridge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-bridge-%j.out
#SBATCH --error=logs/mov-dream-bridge-%j.out
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
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite.yaml}"
GENERATION_PROMPTS="${GENERATION_PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
EDIT_MANIFEST="${EDIT_MANIFEST:-}"
OUT="${OUT:-output/dreamlite_image_bridge/${SLURM_JOB_ID:-local}}"
TARGET_STEP="${TARGET_STEP:-100000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RESUME="${RESUME:-auto}"
STOP_FILE="${TMPDIR%/}/dreamlite_bridge_setup_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs "${OUT}"
if [[ ! -x "${TORCHRUN_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${CONDA_ENV}" >&2
  exit 1
fi
if [[ ! -f "${GENERATION_PROMPTS}" ]]; then
  echo "Missing generation prompts: ${GENERATION_PROMPTS}" >&2
  exit 1
fi
if [[ -n "${EDIT_MANIFEST}" && ! -f "${EDIT_MANIFEST}" ]]; then
  echo "Missing edit manifest: ${EDIT_MANIFEST}" >&2
  exit 1
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV DreamLite image/edit bridge distillation"
echo "CONFIG=${CONFIG} GENERATION_PROMPTS=${GENERATION_PROMPTS} EDIT_MANIFEST=${EDIT_MANIFEST:-none}"
echo "OUT=${OUT} TARGET_STEP=${TARGET_STEP} NPROC_PER_NODE=${NPROC_PER_NODE} RESUME=${RESUME}"
nvidia-smi || true

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices \
  --interval 5 \
  --tensor-mb 4 \
  --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-bridge-setup-${SLURM_JOB_ID:-local}" &
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

EDIT_ARGS=()
if [[ -n "${EDIT_MANIFEST}" ]]; then
  EDIT_ARGS+=(--edit-manifest "${EDIT_MANIFEST}" --edit-probability "${EDIT_PROBABILITY:-0.25}")
fi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config "${CONFIG}" \
  --generation-prompts "${GENERATION_PROMPTS}" \
  "${EDIT_ARGS[@]}" \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume "${RESUME}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-4e-5}" \
  --dtype bf16 \
  --height "${TRAIN_HEIGHT:-512}" \
  --width "${TRAIN_WIDTH:-512}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --functional-start-step "${FUNCTIONAL_START_STEP:-10001}" \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-1}" \
  --closed-loop-start-step "${CLOSED_LOOP_START_STEP:-25001}" \
  --closed-loop-every "${CLOSED_LOOP_EVERY:-4}" \
  --closed-loop-batch-size "${CLOSED_LOOP_BATCH_SIZE:-1}" \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-10000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
