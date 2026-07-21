#!/usr/bin/env bash
#SBATCH --job-name=neo-exp5-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="configs/mobile_ov_neodragon.yaml"
SOURCE_MANIFEST="data/neodragon_openvid100/manifest.csv"
LATENT_DIR="output/neo_exp5_2gpu_smoke/${SLURM_JOB_ID}/latents"
OUT="output/neo_exp5_2gpu_smoke/${SLURM_JOB_ID}/train"
BRIDGE_CKPT="checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"

mkdir -p logs "${LATENT_DIR}" "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/3] Encode eight real OpenVid clips on two GPUs."
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/data_prepare/encode_neodragon_vae_latents.py \
  --config "${CONFIG}" \
  --manifest "${SOURCE_MANIFEST}" \
  --output-dir "${LATENT_DIR}" \
  --max-samples 8

COMMON_ARGS=(
  --config "${CONFIG}"
  --manifest "${LATENT_DIR}/latent_manifest.csv"
  --bridge-ckpt "${BRIDGE_CKPT}"
  --output-dir "${OUT}"
  --steps 6
  --phase-a-steps 2
  --phase-b-steps 2
  --dit-warmup-steps 1
  --bridge-warmup-steps 1
  --bridge-cooldown-steps 1
  --final-cooldown-steps 1
  --batch-size 1
  --max-samples 8
  --parallel fsdp
  --preservation-every 1
  --bridge-functional-every 1
  --diagnostic-every 1
  --log-every 1
  --save-latest-every 4
  --save-archive-every 20
)

echo "[2/3] Run phases A and B, then checkpoint both FSDP optimizers and RNG state."
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_neodragon_exp5.py "${COMMON_ARGS[@]}" \
  --resume none \
  --max-run-steps 4

echo "[3/3] Resume automatically and finish phase C."
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_neodragon_exp5.py "${COMMON_ARGS[@]}" \
  --resume auto

test -f "${OUT}/neodragon_exp5_latest.pt"
test -f "${OUT}/neodragon_exp5_final.pt"
test "$(python -c "import torch; print(torch.load('${OUT}/neodragon_exp5_latest.pt', map_location='cpu', weights_only=False)['step'])")" = "6"
echo "Exp5 two-GPU FSDP train-and-resume smoke test passed: ${OUT}"
