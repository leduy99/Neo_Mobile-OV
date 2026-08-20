#!/usr/bin/env bash
#SBATCH --job-name=neo-pfm-data
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/neo-pfm-data-%j.out
#SBATCH --error=logs/neo-pfm-data-%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-data/neodragon_openvid100/manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/neodragon_flow_ablation_latents}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

cd "${ROOT}"
mkdir -p logs "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" tools/data_prepare/encode_neodragon_vae_latents.py \
  --config configs/mobile_ov_neodragon.yaml \
  --manifest "${SOURCE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-frames 49 \
  --height 320 \
  --width 512 \
  --target-fps 24 \
  --latent-dtype bf16

test -s "${OUTPUT_DIR}/latent_manifest.csv"
wc -l "${OUTPUT_DIR}/latent_manifest.csv"
