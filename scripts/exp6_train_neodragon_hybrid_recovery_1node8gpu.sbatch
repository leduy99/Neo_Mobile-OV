#!/usr/bin/env bash
#SBATCH --job-name=neo-exp6-map
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-exp6-map-%j.out
#SBATCH --error=logs/neo-exp6-map-%j.out
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
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
OUT="${OUT:-output/neo_exp6_hybrid_recovery/${SLURM_JOB_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -z "${BRIDGE_CKPT:-}" ]]; then
  BRIDGE_CANDIDATES=(
    "output/neo_exp1_rollout_64k_to100k/17174692/neodragon_rollout_bridge_latest.pt"
    "output/neo_exp1_rollout_64k_to100k/neodragon_rollout_bridge_latest.pt"
    "checkpoints/hf_mobile_ov/neo_exp1_rollout_64k_to100k/17174692/neodragon_rollout_bridge_latest.pt"
  )
  BRIDGE_CKPT="${BRIDGE_CANDIDATES[0]}"
  for candidate in "${BRIDGE_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
      BRIDGE_CKPT="${candidate}"
      break
    fi
  done
fi

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${MANIFEST}" "${BRIDGE_CKPT}" "${TORCHRUN_BIN}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

echo "Exp6: recover NeoDragon Hybrid's 6x3 one-step transition maps"
echo "MANIFEST=${MANIFEST}"
echo "BRIDGE_CKPT=${BRIDGE_CKPT}"
echo "OUT=${OUT} RESUME=${RESUME:-auto}"
echo "STEPS=${STEPS:-20000} PARITY_STEPS=${PARITY_STEPS:-500} MAP_END_STEP=${MAP_END_STEP:-5000}"
echo "GLOBAL_BATCH=$((NPROC_PER_NODE * ${BATCH_SIZE:-1}))"
echo "LR middle/edge/io=${MIDDLE_LR:-1e-6}/${EDGE_LR:-2.5e-7}/${IO_LR:-5e-7}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_hybrid_recovery.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}" \
  --resume "${RESUME:-auto}" \
  --steps "${STEPS:-20000}" \
  --parity-steps "${PARITY_STEPS:-500}" \
  --map-end-step "${MAP_END_STEP:-5000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --middle-lr "${MIDDLE_LR:-1e-6}" \
  --edge-lr "${EDGE_LR:-2.5e-7}" \
  --io-lr "${IO_LR:-5e-7}" \
  --map-weight "${MAP_WEIGHT:-1.0}" \
  --map-cos-weight "${MAP_COS_WEIGHT:-0.05}" \
  --trust-weight "${TRUST_WEIGHT:-0.15}" \
  --real-endpoint-weight "${REAL_ENDPOINT_WEIGHT:-0.05}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-1000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-5000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
