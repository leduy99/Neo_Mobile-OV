#!/usr/bin/env bash
#SBATCH --job-name=mov-data-anchor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-data-anchor-%j.out
#SBATCH --error=logs/mov-data-anchor-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
PROMPTS="${PROMPTS:-data/mobileov_data_v1/prompt_banks/anchor_teacher_100k.csv}"
OUT="${OUT:-data/mobileov_anchor_teacher_v1}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
IMAGE_HF_REPO="${IMAGE_HF_REPO:-Amshaker/Mobile-OV}"
IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/${IMAGE_HF_FILE}}"
STOP_FILE="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}/mov_data_anchor_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${PROMPTS}"
mkdir -p logs "${OUT}" "$(dirname "${IMAGE_CKPT}")"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-720}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mobileov-anchor-data-setup-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
if [[ ! -s "${IMAGE_CKPT}" ]]; then
  hf download "${IMAGE_HF_REPO}" "${IMAGE_HF_FILE}" --local-dir checkpoints/hf_mobile_ov
fi
test -s "${IMAGE_CKPT}"
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${IMAGE_CONFIG}"
cleanup_heartbeat
trap - EXIT INT TERM

echo "MobileOV-Data-v1 deployment-matched teacher generation"
echo "PROMPTS=${PROMPTS} OUT=${OUT} MAX_SAMPLES=${MAX_SAMPLES}"
echo "DreamLite=V11-balanced; teacher=native NeoDragon monolithic; pair=same exact anchor"
nvidia-smi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/prepare_mobileov_anchor_teacher_data.py \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --image-config "${IMAGE_CONFIG}" \
  --image-bridge-checkpoint "${IMAGE_CKPT}" \
  --video-config "${VIDEO_CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --max-samples "${MAX_SAMPLES}" \
  --dtype "${DTYPE:-bf16}" \
  --phase "${PHASE:-all}" \
  ${EXTRA_ARGS:-}

test -s "${OUT}/manifest.csv"
test -s "${OUT}/metadata.json"
wc -l "${OUT}/manifest.csv"
cat "${OUT}/metadata.json"
