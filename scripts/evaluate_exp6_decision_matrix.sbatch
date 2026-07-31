#!/usr/bin/env bash
#SBATCH --job-name=exp6-matrix
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/exp6_decision_matrix_20260731/logs/%x_%j.out
#SBATCH --error=output/exp6_decision_matrix_20260731/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
OUT="${OUT:-output/exp6_decision_matrix_20260731}"
EXP6="${EXP6:-checkpoints/hf_mobile_ov/neo_exp6_hybrid_recovery/exp6_v1_pilot/neodragon_exp6_step002000.pt}"
BRIDGE80="${BRIDGE80:-checkpoints/hf_mobile_ov/neo_exp1_rollout_64k_to100k/17174692/neodragon_rollout_bridge_latest.pt}"
BRIDGE64="${BRIDGE64:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

mkdir -p "${OUT}/logs"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${EXP6}" "${BRIDGE80}" "${BRIDGE64}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing checkpoint: ${required}" >&2
    exit 1
  fi
done

echo "Exp6 Decision Matrix A-E"
echo "EXP6=${EXP6}"
echo "BRIDGE80=${BRIDGE80}"
echo "BRIDGE64=${BRIDGE64}"
echo "OUT=${OUT} MODE=${MODE:-all} MAX_PROMPTS=${MAX_PROMPTS:--1}"
nvidia-smi

python tools/evaluate_exp6_decision_matrix.py \
  --mode "${MODE:-all}" \
  --exp6 "${EXP6}" \
  --bridge80 "${BRIDGE80}" \
  --bridge64 "${BRIDGE64}" \
  --output-dir "${OUT}" \
  --max-prompts "${MAX_PROMPTS:--1}" \
  --condition-batch-size "${CONDITION_BATCH_SIZE:-4}" \
  --local-batch-size "${LOCAL_BATCH_SIZE:-2}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 2 -type f -printf '%p\t%k KiB\n' | sort
