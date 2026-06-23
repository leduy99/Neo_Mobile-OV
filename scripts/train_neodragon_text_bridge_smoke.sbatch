#!/usr/bin/env bash
#SBATCH --job-name=mov-neo-bridge
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neodragon_text_bridge_smoke/logs/%x_%j.out
#SBATCH --error=output/neodragon_text_bridge_smoke/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
OUT="${ROOT}/output/neodragon_text_bridge_smoke"
PROMPTS="${OUT}/smoke_prompts.txt"
mkdir -p "${OUT}/logs"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
cat > "${PROMPTS}" <<'EOF'
A red fox walking through gentle snowfall.
A cozy Japanese alley at night after rain.
EOF

date
nvidia-smi

python tools/train_neodragon_text_bridge.py \
  --config configs/mobile_ov_neodragon.yaml \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}/bridge" \
  --steps "${STEPS:-3}" \
  --batch-size 1 \
  --log-every 1 \
  --save-every 3

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
date
