#!/usr/bin/env bash
#SBATCH --job-name=neo-exp6-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
MANIFEST="${MANIFEST:-output/neodragon_rollout_smoke/data/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_rollout_64k_to100k/17174692/neodragon_rollout_bridge_latest.pt}"
OUT="${OUT:-output/neo_exp6_hybrid_recovery_smoke/${SLURM_JOB_ID}}"

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  tools/train_neodragon_hybrid_recovery.py \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --output-dir "${OUT}" \
  --resume none \
  --steps "${STEPS:-2}" \
  --parity-steps "${PARITY_STEPS:-1}" \
  --map-end-step "${MAP_END_STEP:-2}" \
  --batch-size 1 \
  --max-samples 8 \
  --no-save-optimizer \
  --no-save-final \
  --log-every 1 \
  --save-latest-every "${SAVE_LATEST_EVERY:-1000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-1000}" \
  ${EXTRA_ARGS:-}
