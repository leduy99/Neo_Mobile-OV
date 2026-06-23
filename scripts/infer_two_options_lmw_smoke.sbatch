#!/usr/bin/env bash
#SBATCH --job-name=newmov-infer2
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
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

export PYTHONPATH="$ROOT:$ROOT/third_party:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VIDEO_PATH="${VIDEO_PATH:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/raw/parts/part_0111/mnt/bn/videodataset-uswest/VDiT/dataset/panda-ours/_QGB6YE-aXs_35_96to231.mp4}"
PROMPT="${PROMPT:-A short cinematic video of a person moving naturally.}"
OUT="${OUT:-output/infer_two_options_lmw_smoke_${SLURM_JOB_ID:-local}}"

nvidia-smi
python tools/infer_two_options_lmw.py \
  --video-path "$VIDEO_PATH" \
  --prompt "$PROMPT" \
  --output-dir "$OUT" \
  --quick-train-steps "${QUICK_TRAIN_STEPS:-5}" \
  --hidden-dim "${LMW_HIDDEN_DIM:-128}" \
  --blocks "${LMW_BLOCKS:-2}" \
  ${EXTRA_ARGS:-}
