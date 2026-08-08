#!/usr/bin/env bash
#SBATCH --job-name=neo-opsd-smoke
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
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
OUT="${OUT:-output/neo_opsd_joint_smoke/${SLURM_JOB_ID:-local}}"

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
  tools/train_neodragon_opsd_joint.py \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --bridge-expected-step 64000 \
  --output-dir "${OUT}" \
  --resume none \
  --steps "${STEPS:-2}" \
  --batch-size 1 \
  --max-samples 8 \
  --lr-warmup-steps 1 \
  --bridge-hold-steps 0 \
  --bridge-ramp-steps 1 \
  --log-every 1 \
  --save-latest-every 1000 \
  --save-archive-every 1000 \
  --no-save-optimizer \
  --no-save-final \
  ${EXTRA_ARGS:-}
