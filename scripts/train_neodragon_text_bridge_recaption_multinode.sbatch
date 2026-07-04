#!/usr/bin/env bash
#SBATCH --job-name=neo-recap-distill
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
if [[ ! -d "${ROOT}/new_mobile_ov" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT}"
mkdir -p logs output/neodragon_text_bridge_recaption_distill

if [[ -z "${NEO_ROOT:-}" ]]; then
  if [[ -d "${ROOT}/../neodragon" ]]; then
    NEO_ROOT="$(cd "${ROOT}/../neodragon" && pwd)"
  else
    NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
  fi
fi
if [[ ! -d "${NEO_ROOT}" ]]; then
  echo "Missing Neodragon repo: NEO_ROOT=${NEO_ROOT}" >&2
  echo "Set NEO_ROOT=/path/to/neodragon before sbatch." >&2
  exit 1
fi

if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -d "/share_4/users/duy/.conda/envs/neo_mobileov" ]]; then
    CONDA_ENV="/share_4/users/duy/.conda/envs/neo_mobileov"
  elif [[ -d "/proj/cvl/users/x_fahkh2/envs/neo_mobileov" ]]; then
    CONDA_ENV="/proj/cvl/users/x_fahkh2/envs/neo_mobileov"
  else
    CONDA_ENV="/proj/cvl/users/x_fahkh2/envs/mobileov"
  fi
fi

if [[ -f "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f "/share_0/conda/etc/profile.d/conda.sh" ]]; then
  source /share_0/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
elif [[ -x "${CONDA_ENV}/bin/python" ]]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  echo "Could not activate CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi
if [[ -d "${CONDA_ENV}/bin" ]]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
if [[ -x "${CONDA_ENV}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python"
fi
if [[ -x "${CONDA_ENV}/bin/torchrun" ]]; then
  TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
fi

LOCAL_OPENVID_ROOT="${LOCAL_OPENVID_ROOT:-${ROOT}/download_data/data/openvid}"
if [[ -z "${RECAPTION_CSV:-}" ]]; then
  if [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions_merged.csv" ]]; then
    RECAPTION_CSV="${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions_merged.csv"
  elif [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions.csv" ]]; then
    RECAPTION_CSV="${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions.csv"
  elif [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all.csv" ]]; then
    RECAPTION_CSV="${LOCAL_OPENVID_ROOT}/manifests/openvid_all.csv"
  else
    RECAPTION_CSV="${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions_merged.csv"
  fi
fi
if [[ ! -f "${RECAPTION_CSV}" ]]; then
  echo "Missing recaption CSV: ${RECAPTION_CSV}" >&2
  echo "Run first: bash scripts/import_openvid_data_to_repo.sh" >&2
  echo "Override with: RECAPTION_CSV=/path/to/openvid_all_recaptions.csv sbatch $0" >&2
  exit 1
fi

NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29577}"
RDZV_ID="${RDZV_ID:-neo_recap_distill_${SLURM_JOB_ID:-local}}"

PARALLEL="${PARALLEL:-fsdp}"
OUT="${OUT:-${ROOT}/output/neodragon_text_bridge_recaption_distill/${PARALLEL}_${SLURM_JOB_ID:-local}}"
LATEST="${LATEST:-${ROOT}/output/neodragon_text_bridge_recaption_distill/${PARALLEL}_latest/bridge}"
mkdir -p "${OUT}/bridge" "${LATEST}"

export ROOT NEO_ROOT RECAPTION_CSV OUT LATEST
export PYTHON_BIN TORCHRUN_BIN
export NNODES GPUS_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT RDZV_ID
export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export HF_HOME="${HF_HOME:-${ROOT}/output/cache/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/output/cache/torch_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/output/cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOBILEOV_CAPTION_AUG="${MOBILEOV_CAPTION_AUG:-1}"
export MOBILEOV_CAPTION_AUG_COLUMNS="${MOBILEOV_CAPTION_AUG_COLUMNS:-caption_short,caption_medium,caption_long}"
export MOBILEOV_CAPTION_AUG_WEIGHTS="${MOBILEOV_CAPTION_AUG_WEIGHTS:-1,1,1}"
export MOBILEOV_CAPTION_FALLBACK_COLUMN="${MOBILEOV_CAPTION_FALLBACK_COLUMN:-caption}"

echo "ROOT=${ROOT}"
echo "NEO_ROOT=${NEO_ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "TORCHRUN_BIN=${TORCHRUN_BIN}"
echo "RECAPTION_CSV=${RECAPTION_CSV}"
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"
echo "PARALLEL=${PARALLEL} OUT=${OUT}"
echo "Caption aug: enabled=${MOBILEOV_CAPTION_AUG} columns=${MOBILEOV_CAPTION_AUG_COLUMNS} weights=${MOBILEOV_CAPTION_AUG_WEIGHTS}"
date
nvidia-smi || true

srun --ntasks="${NNODES}" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" bash -c '
set -euo pipefail
cd "${ROOT}"
echo "node=$(hostname) node_rank=${SLURM_NODEID:-0} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${SLURM_NODEID:-0}" \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  tools/train_neodragon_text_bridge.py \
  --config "${CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --prompts "${RECAPTION_CSV}" \
  --output-dir "${OUT}/bridge" \
  --steps "${STEPS:-10000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${LR:-1e-4}" \
  --max-prompts "${MAX_PROMPTS:--1}" \
  --parallel "${PARALLEL}" \
  --deepspeed-zero-stage "${DEEPSPEED_ZERO_STAGE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-1}" \
  --log-every "${LOG_EVERY:-20}" \
  --save-every "${SAVE_EVERY:-500}" \
  --caption-variant-columns "${MOBILEOV_CAPTION_AUG_COLUMNS}" \
  --caption-variant-weights "${MOBILEOV_CAPTION_AUG_WEIGHTS}" \
  --caption-fallback-column "${MOBILEOV_CAPTION_FALLBACK_COLUMN}" \
  ${CAPTION_AUG_FLAG:-} \
  ${EXTRA_ARGS:-}
'

cp -f "${OUT}/bridge/neodragon_text_bridge_latest.pt" "${LATEST}/neodragon_text_bridge_latest.pt"
cp -f "${OUT}/bridge/history.json" "${LATEST}/history.json"
if [[ -f "${OUT}/bridge/deepspeed_config.json" ]]; then
  cp -f "${OUT}/bridge/deepspeed_config.json" "${LATEST}/deepspeed_config.json"
fi

find "${OUT}" -maxdepth 3 -type f -print -exec ls -lh {} \;
find "${LATEST}" -maxdepth 1 -type f -print -exec ls -lh {} \;
date
