#!/usr/bin/env bash
#SBATCH --job-name=dream-v3-smoke
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
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
OUT="${OUT:-output/dreamlite_compact_v3_2gpu_smoke/${SLURM_JOB_ID:-local}}"

mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_dreamlite_image_bridge.py \
  --config configs/mobile_ov_dreamlite_compact_v3.yaml \
  --generation-prompts configs/prompts/dreamlite_smoke.csv \
  --output-dir "${OUT}" \
  --target-step 3 \
  --resume none \
  --batch-size 1 \
  --max-samples 2 \
  --height 256 \
  --width 256 \
  --representation-mode direct \
  --token-norm-weight 0.10 \
  --variance-weight 0.25 \
  --token-mean-weight 0.25 \
  --token-std-weight 0.25 \
  --projected-weight 1.0 \
  --functional-start-step 2 \
  --functional-ramp-steps 1 \
  --functional-batch-size 1 \
  --closed-loop-start-step 3 \
  --closed-loop-ramp-steps 1 \
  --closed-loop-every 1 \
  --closed-loop-batch-size 1 \
  --log-every 1 \
  --save-latest-every 2 \
  --save-archive-every 3

test -s "${OUT}/dreamlite_image_bridge_latest.pt"
test -s "${OUT}/history.jsonl"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/infer_dreamlite_image_bridge.py \
  --config configs/mobile_ov_dreamlite_compact_v3.yaml \
  --bridge-checkpoint "${OUT}/dreamlite_image_bridge_latest.pt" \
  --prompt "A red panda reads a small book beside a window." \
  --output "${OUT}/inference.png" \
  --height 256 \
  --width 256 \
  --steps 4 \
  --seed 42
test -s "${OUT}/inference.png"
echo "DreamLite compact V3 two-GPU smoke passed: ${OUT}"
