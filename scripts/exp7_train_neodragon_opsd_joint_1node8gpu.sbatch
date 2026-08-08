#!/usr/bin/env bash
#SBATCH --job-name=neo-opsd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/neo-opsd-%j.out
#SBATCH --error=logs/neo-opsd-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
MANIFEST="${MANIFEST:-data/openvid_neodragon_2s_latents/latent_manifest.csv}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-local}}"
OUT="${OUT:-output/neo_opsd_joint/${RUN_NAME}}"

if [[ -z "${BRIDGE_CKPT:-}" ]]; then
  BRIDGE_CANDIDATES=(
    "output/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
    "checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt"
  )
  BRIDGE_CKPT="${BRIDGE_CANDIDATES[0]}"
  for candidate in "${BRIDGE_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
      BRIDGE_CKPT="${candidate}"
      break
    fi
  done
fi

mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${MANIFEST}" "${BRIDGE_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

BRIDGE_STEP="$(${PYTHON_BIN} - "${BRIDGE_CKPT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
print(int(payload.get("step", -1)))
PY
)"
if [[ "${BRIDGE_STEP}" != "64000" ]]; then
  echo "OPSD-Neo must initialize from Exp1-64K; found step=${BRIDGE_STEP}." >&2
  exit 1
fi
BRIDGE_SHA256="$(sha256sum "${BRIDGE_CKPT}" | awk '{print $1}')"

echo "OPSD-Neo joint fine-tuning"
echo "MANIFEST=${MANIFEST}"
echo "BRIDGE_CKPT=${BRIDGE_CKPT} STEP=${BRIDGE_STEP} SHA256=${BRIDGE_SHA256}"
echo "RUN_NAME=${RUN_NAME} OUT=${OUT} RESUME=${RESUME:-auto}"
echo "STEPS=${STEPS:-20000} GLOBAL_BATCH=$((NPROC_PER_NODE * ${BATCH_SIZE:-1}))"
echo "DiT LR middle/edge/io=${DIT_MIDDLE_LR:-2e-7}/${DIT_EDGE_LR:-5e-8}/${DIT_IO_LR:-1e-7}"
echo "Bridge LR=${BRIDGE_LR:-2e-6} hold/ramp=${BRIDGE_HOLD_STEPS:-1000}/${BRIDGE_RAMP_STEPS:-2000}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_neodragon_opsd_joint.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --bridge-expected-step 64000 \
  --bridge-sha256 "${BRIDGE_SHA256}" \
  --output-dir "${OUT}" \
  --resume "${RESUME:-auto}" \
  --steps "${STEPS:-20000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --dit-middle-lr "${DIT_MIDDLE_LR:-2e-7}" \
  --dit-edge-lr "${DIT_EDGE_LR:-5e-8}" \
  --dit-io-lr "${DIT_IO_LR:-1e-7}" \
  --bridge-lr "${BRIDGE_LR:-2e-6}" \
  --bridge-hold-steps "${BRIDGE_HOLD_STEPS:-1000}" \
  --bridge-ramp-steps "${BRIDGE_RAMP_STEPS:-2000}" \
  --opsd-weight "${OPSD_WEIGHT:-1.0}" \
  --velocity-cos-weight "${VELOCITY_COS_WEIGHT:-0.05}" \
  --trust-weight "${TRUST_WEIGHT:-0.25}" \
  --bridge-anchor-weight "${BRIDGE_ANCHOR_WEIGHT:-0.02}" \
  --early-preserve-weight "${EARLY_PRESERVE_WEIGHT:-0.10}" \
  --teacher-gate-margin "${TEACHER_GATE_MARGIN:-0.0}" \
  --teacher-gate-ramp "${TEACHER_GATE_RAMP:-0.05}" \
  --teacher-gate-check-step "${TEACHER_GATE_CHECK_STEP:-120}" \
  --min-active-teacher-fraction "${MIN_ACTIVE_TEACHER_FRACTION:-0.10}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --caption-fallback-column caption \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-1000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-5000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
