#!/usr/bin/env bash
#SBATCH --job-name=neo-ckpt-test
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neo_checkpoint_test/logs/%x_%j.out
#SBATCH --error=output/neo_checkpoint_test/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
EXP1_CHECKPOINT="${EXP1_CHECKPOINT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
EXP2_CHECKPOINT="${EXP2_CHECKPOINT:-checkpoints/hf_mobile_ov/neo_exp2_joint_flow_distill/17108894/neodragon_dit_bridge_latest.pt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/neo_latest_checkpoints_test}"

source /share_0/conda/etc/profile.d/conda.sh
conda activate /share_4/users/duy/.conda/envs/neo_mobileov

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROMPTS=(
  "A red fox walking through gentle snowfall, cinematic wildlife footage."
  "A golden retriever running through a sunlit field of wildflowers."
)
CHECKPOINTS=("${EXP1_CHECKPOINT}" "${EXP2_CHECKPOINT}")
TAGS=("exp1_step64k" "exp2_step40k")

cd "${ROOT}"
nvidia-smi

for checkpoint_index in "${!CHECKPOINTS[@]}"; do
  for prompt_index in "${!PROMPTS[@]}"; do
    python tools/infer_mobile_ov_neodragon.py \
      --config configs/mobile_ov_neodragon.yaml \
      --prompt "${PROMPTS[$prompt_index]}" \
      --output-dir "${OUT_ROOT}/${TAGS[$checkpoint_index]}/prompt_$((prompt_index + 1))" \
      --condition-source bridge \
      --bridge-ckpt "${CHECKPOINTS[$checkpoint_index]}" \
      --seed 1234 \
      --profile
  done
done

find "${OUT_ROOT}" -type f -print -exec ls -lh {} \;
