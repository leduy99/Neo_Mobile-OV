#!/usr/bin/env bash
# Jointly fine-tune the Mobile-OV text bridge and released NeoDragon multistep
# DiT on precomputed OpenVid latents. The bridge is protected at the beginning,
# then updated with direct and CFG-aware frozen-teacher anchors.
#
# Default: use the latest validated monolithic video-units bridge from job
# 17318011. BRIDGE_CKPT can still override it for a controlled ablation.
#SBATCH --job-name=neo-mono-joint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-mono-joint-%j.out
#SBATCH --error=logs/neo-mono-joint-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
DEFAULT_BRIDGE_CKPT="output/neo_monolithic_video_units_text_bridge/monolithic_video_units_exp1_functional_17318011/neodragon_text_bridge_latest.pt"
BRIDGE_CKPT="${BRIDGE_CKPT:-${DEFAULT_BRIDGE_CKPT}}"
RUN_NAME="${RUN_NAME:-mobileov_monolithic_joint_flow_${SLURM_JOB_ID}}"
OUT="${OUT:-output/mobileov_monolithic_joint_flow/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

test -f "${MANIFEST}"
test -f "${BRIDGE_CKPT}"
test -x "${TORCHRUN_BIN}"
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV joint bridge + NeoDragon monolithic DiT fine-tuning"
echo "MANIFEST=${MANIFEST} BRIDGE_CKPT=${BRIDGE_CKPT} OUT=${OUT}"
echo "Bridge is detached through step ${BRIDGE_START_STEP:-5000}; trainable weights use FP32 masters."
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_dit_bridge.py \
  --config "${CONFIG}" \
  --target-stack multistep \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}" \
  --steps "${STEPS:-50000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${DIT_LR:-1e-6}" \
  --bridge-lr "${BRIDGE_LR:-2e-6}" \
  --train-bridge \
  --bridge-start-step "${BRIDGE_START_STEP:-5000}" \
  --bridge-warmup-steps "${BRIDGE_WARMUP_STEPS:-2000}" \
  --train-last-n-blocks 0 \
  --parallel fsdp \
  --dit-trainable-fp32 \
  --bridge-trainable-fp32 \
  --use-precomputed-latents \
  --caption-aug \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --caption-fallback-column caption \
  --flow-start-weight "${FLOW_START_WEIGHT:-0.2}" \
  --flow-weight "${FLOW_WEIGHT:-1.0}" \
  --flow-final-weight "${FLOW_FINAL_WEIGHT:-0.5}" \
  --flow-ramp-steps "${FLOW_RAMP_STEPS:-2000}" \
  --cooldown-steps "${COOLDOWN_STEPS:-10000}" \
  --dit-warmup-steps "${DIT_WARMUP_STEPS:-1000}" \
  --distill-weight "${DISTILL_WEIGHT:-0.5}" \
  --distill-cos-weight "${DISTILL_COS_WEIGHT:-0.05}" \
  --preservation-weight "${PRESERVATION_WEIGHT:-0.25}" \
  --preservation-cos-weight "${PRESERVATION_COS_WEIGHT:-0.025}" \
  --preservation-every "${PRESERVATION_EVERY:-4}" \
  --bridge-repr-weight "${BRIDGE_REPR_WEIGHT:-0.5}" \
  --bridge-repr-final-weight "${BRIDGE_REPR_FINAL_WEIGHT:-0.25}" \
  --bridge-raw-token-weight 0.25 \
  --bridge-normalized-token-weight 1.0 \
  --bridge-cos-weight 0.5 \
  --bridge-token-norm-weight 0.1 \
  --bridge-pooled-weight 0.25 \
  --bridge-pooled-cos-weight 0.2 \
  --bridge-relational-weight 0.1 \
  --bridge-functional-weight "${BRIDGE_FUNCTIONAL_WEIGHT:-0.5}" \
  --bridge-functional-cos-weight "${BRIDGE_FUNCTIONAL_COS_WEIGHT:-0.05}" \
  --bridge-functional-every "${BRIDGE_FUNCTIONAL_EVERY:-4}" \
  --bridge-functional-batch-size 1 \
  --cfg-distill-weight "${CFG_DISTILL_WEIGHT:-0.1}" \
  --cfg-distill-cos-weight "${CFG_DISTILL_COS_WEIGHT:-0.02}" \
  --bridge-cfg-functional-weight "${BRIDGE_CFG_FUNCTIONAL_WEIGHT:-0.1}" \
  --bridge-cfg-functional-cos-weight "${BRIDGE_CFG_FUNCTIONAL_COS_WEIGHT:-0.02}" \
  --cfg-every "${CFG_EVERY:-4}" \
  --cfg-scale "${CFG_SCALE:-5.0}" \
  --diagnostic-every "${DIAGNOSTIC_EVERY:-500}" \
  --log-every "${LOG_EVERY:-20}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --archive-every "${ARCHIVE_EVERY:-10000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
