#!/usr/bin/env bash
#SBATCH --job-name=neo-speed
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neodragon_speed_benchmark/logs/%x_%j.out
#SBATCH --error=output/neodragon_speed_benchmark/logs/%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
if [[ ! -d "${ROOT}/new_mobile_ov" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT}"
mkdir -p output/neodragon_speed_benchmark/logs

if [[ -f "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
elif [[ -f "/share_0/conda/etc/profile.d/conda.sh" ]]; then
  source /share_0/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
elif [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python" ]]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/checkpoints/torch_home}"

date
nvidia-smi || true

python tools/benchmark_neodragon_inference.py \
  --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --bridge-ckpt "${BRIDGE_CKPT:-output/neodragon_text_bridge_smoke/bridge/neodragon_text_bridge_latest.pt}" \
  --prompt "${PROMPT:-A red fox walking through gentle snowfall, cinematic wildlife footage.}" \
  --output-dir "${OUT:-output/neodragon_speed_benchmark/${SLURM_JOB_ID:-local}}" \
  --runs "${RUNS:-2}" \
  --warmup "${WARMUP:-1}" \
  --save-videos

find output/neodragon_speed_benchmark -maxdepth 3 -type f -print -exec ls -lh {} \;
date
