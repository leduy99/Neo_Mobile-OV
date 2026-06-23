#!/usr/bin/env bash
#SBATCH --job-name=mov-neo-inf
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/mobile_ov_neodragon_smoke/logs/%x_%j.out
#SBATCH --error=output/mobile_ov_neodragon_smoke/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
mkdir -p "${ROOT}/output/mobile_ov_neodragon_smoke/logs"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
date
nvidia-smi

CMD=(
  python tools/infer_mobile_ov_neodragon.py
  --config configs/mobile_ov_neodragon.yaml
  --prompt "${PROMPT:-A red fox walking through gentle snowfall, cinematic wildlife footage.}"
  --output-dir output/mobile_ov_neodragon_smoke/generated
  --condition-source "${CONDITION_SOURCE:-native}"
  --profile
)
if [[ -n "${BRIDGE_CKPT:-}" ]]; then
  CMD+=(--bridge-ckpt "${BRIDGE_CKPT}")
fi
"${CMD[@]}"

find output/mobile_ov_neodragon_smoke/generated -maxdepth 1 -type f -print -exec ls -lh {} \;
date
