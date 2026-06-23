#!/usr/bin/env bash
#SBATCH --job-name=neo-bridge-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/neodragon_text_bridge_distill/logs/%x_%j.out
#SBATCH --error=output/neodragon_text_bridge_distill/logs/%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
if [[ ! -d "${ROOT}/new_mobile_ov" ]]; then
  ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
fi
NEO_ROOT="${NEO_ROOT:-/share_4/users/duy/project/unified_video/neodragon}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/neodragon_openvid100}"
OUT="${OUT:-${ROOT}/output/neodragon_text_bridge_distill/${PARALLEL:-fsdp}_${SLURM_JOB_ID:-local}}"

mkdir -p "${OUT}" "${ROOT}/output/neodragon_text_bridge_distill/logs"
cd "${ROOT}"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-${NEO_ROOT}/models/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${NEO_ROOT}/models/torch_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${DATA_DIR}/manifest.csv" ]]; then
  echo "Preparing Neodragon OpenVid100 at ${DATA_DIR}"
  bash scripts/prepare_neodragon_openvid100.sh
else
  echo "Using existing data manifest: ${DATA_DIR}/manifest.csv"
fi

date
nvidia-smi
python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count=", torch.cuda.device_count())
PY

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --prompts "${PROMPTS:-${DATA_DIR}/manifest.csv}" \
  --output-dir "${OUT}/bridge" \
  --steps "${STEPS:-20}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${LR:-1e-4}" \
  --max-prompts "${MAX_PROMPTS:-100}" \
  --parallel "${PARALLEL:-fsdp}" \
  --deepspeed-zero-stage "${DEEPSPEED_ZERO_STAGE:-1}" \
  --log-every "${LOG_EVERY:-1}" \
  --save-every "${SAVE_EVERY:-20}" \
  ${EXTRA_ARGS:-}

LATEST="${ROOT}/output/neodragon_text_bridge_distill/${PARALLEL:-fsdp}_latest/bridge"
mkdir -p "${LATEST}"
cp -f "${OUT}/bridge/neodragon_text_bridge_latest.pt" "${LATEST}/neodragon_text_bridge_latest.pt"
cp -f "${OUT}/bridge/history.json" "${LATEST}/history.json"
if [[ -f "${OUT}/bridge/deepspeed_config.json" ]]; then
  cp -f "${OUT}/bridge/deepspeed_config.json" "${LATEST}/deepspeed_config.json"
fi

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
find "${LATEST}" -maxdepth 1 -type f -print -exec ls -lh {} \;
date
