#!/usr/bin/env bash
#SBATCH --job-name=lmw-feas100
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
mkdir -p logs output data/lmw_feasibility

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="$ROOT:$ROOT/third_party/sana:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/mobile_ov_current.yaml}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv}"
DATA_DIR="${DATA_DIR:-data/lmw_feasibility/openvid100_81f}"
OUT="${OUT:-output/lmw_feasibility_100_${SLURM_JOB_ID:-local}}"
WAN_VAE="${WAN_VAE:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth}"

NUM_VIDEOS="${NUM_VIDEOS:-100}"
OFFSET="${OFFSET:-0}"
FRAME_NUM="${FRAME_NUM:-81}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SAMPLING_RATE="${SAMPLING_RATE:-3}"

LMW_STEPS="${LMW_STEPS:-300}"
LMW_HIDDEN_DIM="${LMW_HIDDEN_DIM:-512}"
LMW_BLOCKS="${LMW_BLOCKS:-10}"
LMW_TEXT_MODE="${LMW_TEXT_MODE:-zero}"
LMW_TEXT_SEED="${LMW_TEXT_SEED:-0}"
DECODE_COUNT="${DECODE_COUNT:-4}"

nvidia-smi

if [[ ! -f "${DATA_DIR}/manifest.csv" ]]; then
  python tools/prepare_wanvae_latents.py \
    --manifest "$SOURCE_MANIFEST" \
    --vae-path "$WAN_VAE" \
    --output-dir "$DATA_DIR" \
    --max-videos "$NUM_VIDEOS" \
    --offset "$OFFSET" \
    --frame-num "$FRAME_NUM" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --sampling-rate "$SAMPLING_RATE" \
    --device cuda
else
  echo "Using existing latent manifest: ${DATA_DIR}/manifest.csv"
fi

python tools/train_lmw_projection.py \
  --config "$CONFIG" \
  --manifest "${DATA_DIR}/manifest.csv" \
  --output-dir "$OUT" \
  --steps "$LMW_STEPS" \
  --hidden-dim "$LMW_HIDDEN_DIM" \
  --blocks "$LMW_BLOCKS" \
  --text-mode "$LMW_TEXT_MODE" \
  --text-seed "$LMW_TEXT_SEED" \
  --device cuda

python tools/evaluate_lmw_projection.py \
  --config "$CONFIG" \
  --checkpoint "${OUT}/lmw_latest.pt" \
  --manifest "${DATA_DIR}/manifest.csv" \
  --output-dir "$OUT" \
  --text-mode checkpoint \
  --vae-path "$WAN_VAE" \
  --decode-count "$DECODE_COUNT" \
  --device cuda
