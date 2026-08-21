#!/usr/bin/env bash
# Adapt the monolithic NeoDragon DiT to the fixed CFG-v2 bridge without moving
# the bridge interface. Training covers external-anchor video units 1..6 only.
#SBATCH --job-name=neo-pfm-cfg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/neo-pfm-cfg-%j.out
#SBATCH --error=logs/neo-pfm-cfg-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-output/neo_monolithic_cfg_bridge_v2/monolithic_cfg_bridge_v2_17327720/neodragon_text_bridge_latest.pt}"
INIT_JOINT_CKPT="${INIT_JOINT_CKPT:-}"
RUN_NAME="${RUN_NAME:-mobileov_monolithic_pyramid_cfg_dit_${SLURM_JOB_ID}}"
OUT="${OUT:-output/mobileov_monolithic_pyramid_cfg_dit/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
STEPS="${STEPS:-1000}"

cd "${ROOT}"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: ${label} is missing or empty: ${path}" >&2
    exit 1
  fi
}

require_file "training manifest" "${MANIFEST}"
require_file "frozen bridge checkpoint" "${BRIDGE_CKPT}"
INIT_ARGS=()
if [[ -n "${INIT_JOINT_CKPT}" ]]; then
  require_file "initial joint checkpoint" "${INIT_JOINT_CKPT}"
  INIT_ARGS=(--init-joint-ckpt "${INIT_JOINT_CKPT}")
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "ERROR: torchrun is missing or not executable: ${TORCHRUN_BIN}" >&2
  exit 1
fi
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "NeoDragon fixed-bridge Pyramidal-Flow + CFG response refinement"
if [[ -n "${INIT_JOINT_CKPT}" ]]; then
  echo "Initialization: continuation from ${INIT_JOINT_CKPT}"
else
  echo "Initialization: clean released NeoDragon multistep DiT + independent frozen bridge"
fi
echo "BRIDGE_CKPT=${BRIDGE_CKPT} MANIFEST=${MANIFEST} OUT=${OUT}"
echo "Contract: bridge frozen, external anchor, video units 1..6, CFG scale=${CFG_SCALE:-5.0}"
echo "world_size=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE} global_batch=$((NPROC_PER_NODE * BATCH_SIZE)) steps=${STEPS}"
nvidia-smi || true

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  --config "${CONFIG}" \
  --target-stack multistep \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  "${INIT_ARGS[@]}" \
  --output-dir "${OUT}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-samples "${MAX_SAMPLES}" \
  --use-precomputed-latents \
  --train-last-n-blocks 0 \
  --parallel fsdp \
  --dit-trainable-fp32 \
  --no-train-bridge \
  --flow-contract pyramid \
  --stage-sampling-ratio 1,1,1 \
  --history-corrupt-max "${HISTORY_CORRUPT_MAX:-0.3333333333333333}" \
  --flow-weight "${FLOW_WEIGHT:-1.0}" \
  --distill-weight "${DISTILL_WEIGHT:-1.0}" \
  --distill-cos-weight "${DISTILL_COS_WEIGHT:-0.1}" \
  --cfg-distill-weight "${CFG_DISTILL_WEIGHT:-0.5}" \
  --cfg-distill-cos-weight "${CFG_DISTILL_COS_WEIGHT:-0.1}" \
  --cfg-every "${CFG_EVERY:-1}" \
  --cfg-scale "${CFG_SCALE:-5.0}" \
  --preservation-weight "${PRESERVATION_WEIGHT:-0.5}" \
  --preservation-cos-weight "${PRESERVATION_COS_WEIGHT:-0.05}" \
  --preservation-every 1 \
  --lr "${DIT_LR:-3e-6}" \
  --dit-warmup-steps "${DIT_WARMUP_STEPS:-100}" \
  --clip-grad-norm "${CLIP_GRAD_NORM:-1.0}" \
  --diagnostic-every "${DIAGNOSTIC_EVERY:-50}" \
  --log-every "${LOG_EVERY:-10}" \
  --save-every "${SAVE_EVERY:-100}" \
  --archive-every "${ARCHIVE_EVERY:-250}" \
  ${EXTRA_ARGS:-}

echo "Completed fixed-bridge CFG DiT refinement: ${OUT}"
find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
