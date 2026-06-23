#!/usr/bin/env bash
#SBATCH --job-name=wanvae-sanity100
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/new_mobile_ov" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"
mkdir -p logs output

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="$ROOT:$ROOT/third_party/sana:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MANIFEST="${MANIFEST:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv}"
OUT="${OUT:-output/wanvae_anchor_sanity_100_${SLURM_JOB_ID:-local}}"

nvidia-smi
python tools/wanvae_anchor_sanity_checks.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT" \
  --num-videos "${NUM_VIDEOS:-100}" \
  --offset "${OFFSET:-0}" \
  --frame-num "${FRAME_NUM:-81}" \
  --frame-idx "${FRAME_IDX:-10}" \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-832}" \
  --sampling-rate "${SAMPLING_RATE:-3}" \
  --dtype "${DTYPE:-bf16}" \
  ${EXTRA_ARGS:-}
