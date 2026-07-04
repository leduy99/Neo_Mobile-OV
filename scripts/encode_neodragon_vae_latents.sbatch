#!/usr/bin/env bash
#SBATCH --job-name=neo-vae-enc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

# Simple public entrypoint:
#   sbatch scripts/encode_neodragon_vae_latents.sbatch
#
# For a smaller smoke test:
#   sbatch --gres=gpu:2 --cpus-per-task=8 --mem=96G \
#     --export=ALL,NPROC_PER_NODE=2,MAX_SAMPLES=4,OUT=output/neo_vae_2gpu_smoke/latents \
#     scripts/encode_neodragon_vae_latents.sbatch

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
if [[ ! -x "${CONDA_ENV}/bin/torchrun" && -x "/share_4/users/duy/.conda/envs/neo_mobileov/bin/torchrun" ]]; then
  CONDA_ENV="/share_4/users/duy/.conda/envs/neo_mobileov"
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"

CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
CLIP_MANIFEST="${CLIP_MANIFEST:-data/openvid_neodragon_2s/manifest.csv}"
OUT="${OUT:-data/openvid_neodragon_2s_latents}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LATENT_DTYPE="${LATENT_DTYPE:-bf16}"
PREPARE_CLIPS="${PREPARE_CLIPS:-1}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS=',' read -r -a _cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_cuda_devices[@]}"
  else
    NPROC_PER_NODE="8"
  fi
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "Invalid NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

mkdir -p logs "${OUT}"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun: ${TORCHRUN_BIN}" >&2
  exit 1
fi

if [[ "${PREPARE_CLIPS}" == "1" && ! -f "${CLIP_MANIFEST}" ]]; then
  echo "Preparing clip manifest because missing: ${CLIP_MANIFEST}"
  OUT_DIR="$(dirname "${CLIP_MANIFEST}")" bash scripts/prepare_openvid_neodragon_2s.sh
fi
if [[ ! -f "${CLIP_MANIFEST}" ]]; then
  echo "Missing CLIP_MANIFEST=${CLIP_MANIFEST}" >&2
  echo "Run first:" >&2
  echo "  bash scripts/copy_openvid_captions_to_repo.sh" >&2
  echo "  bash scripts/download_openvid_raw_to_repo.sh" >&2
  exit 1
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${PWD}/checkpoints/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${PWD}/checkpoints/torch_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/output/cache/triton}"

echo "PWD=${PWD}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "CLIP_MANIFEST=${CLIP_MANIFEST}"
echo "OUT=${OUT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} MAX_SAMPLES=${MAX_SAMPLES} LATENT_DTYPE=${LATENT_DTYPE}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/data_prepare/encode_neodragon_vae_latents.py \
  --config "${CONFIG}" \
  --manifest "${CLIP_MANIFEST}" \
  --output-dir "${OUT}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-frames 49 \
  --height 320 \
  --width 512 \
  --target-fps 24 \
  --latent-dtype "${LATENT_DTYPE}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'latent_manifest.csv' -o -name 'failed.csv' \) -print -exec ls -lh {} \;
