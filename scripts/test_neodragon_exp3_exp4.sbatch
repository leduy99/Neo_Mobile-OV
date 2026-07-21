#!/usr/bin/env bash
#SBATCH --job-name=neo-exp34-test
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neo_exp3_exp4_test/logs/%x_%j.out
#SBATCH --error=output/neo_exp3_exp4_test/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
EXP3_CHECKPOINT="${EXP3_CHECKPOINT:-checkpoints/hf_mobile_ov/neo_exp3_joint_from_scratch/17104365/neodragon_dit_bridge_latest.pt}"
EXP4_CHECKPOINT="${EXP4_CHECKPOINT:-checkpoints/hf_mobile_ov/neo_exp4_flow_only/17104367/neodragon_dit_bridge_latest.pt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/neo_exp3_exp4_test}"

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
CHECKPOINTS=("${EXP3_CHECKPOINT}" "${EXP4_CHECKPOINT}")
TAGS=("exp3_step80k" "exp4_step100k")

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
