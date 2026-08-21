#!/usr/bin/env bash
#SBATCH --job-name=neo-joint-eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/neo-joint-eval-%j.out
#SBATCH --error=logs/neo-joint-eval-%j.out

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
V1="${V1:-checkpoints/hf_mobile_ov/mobileov_monolithic_joint_flow/mobileov_monolithic_joint_flow_17327721/neodragon_dit_bridge_latest.pt}"
V2="${V2:-checkpoints/hf_mobile_ov/mobileov_monolithic_joint_cfg_bridge_v2/mobileov_monolithic_joint_cfg_bridge_v2_17333147/neodragon_dit_bridge_latest.pt}"
OUT="${OUT:-output/joint_monolithic_v1_v2_controlled_${SLURM_JOB_ID}}"

test -x "${PYTHON_BIN}"
test -f "${V1}"
test -f "${V2}"
mkdir -p logs "${OUT}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/share_4/users/duy/.cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u tools/evaluate_neodragon_joint_monolithic.py \
  --checkpoint "joint_v1=${V1}" \
  --checkpoint "joint_v2=${V2}" \
  --output-dir "${OUT}" \
  --seed "${SEED:-20260820}" \
  --dtype "${DTYPE:-bf16}" \
  ${EXTERNAL_ANCHOR:+--external-anchor}

find "${OUT}" -maxdepth 2 -type f -printf '%p %k KB\n' | sort
