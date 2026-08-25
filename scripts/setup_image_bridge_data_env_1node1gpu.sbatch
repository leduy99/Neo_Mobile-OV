#!/usr/bin/env bash
#SBATCH --job-name=mov-imgdata-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/mov-imgdata-env-%j.out
#SBATCH --error=logs/mov-imgdata-env-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

BASE_ENV="${BASE_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
TARGET_ENV="${TARGET_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_image_data}"
BASE_PYTHON="${BASE_ENV}/bin/python"
STOP_FILE="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}/image_bridge_env_${SLURM_JOB_ID:-local}.stop"

test -x "${BASE_PYTHON}"
mkdir -p logs "$(dirname "${TARGET_ENV}")" "$(dirname "${STOP_FILE}")"

rm -f "${STOP_FILE}"
"${BASE_PYTHON}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "image-bridge-data-env-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -x "${TARGET_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${TARGET_ENV}"
fi

"${TARGET_ENV}/bin/python" -m pip install --upgrade pip
"${TARGET_ENV}/bin/python" -m pip install --upgrade \
  "transformers>=5.2.0,<6" \
  "accelerate>=1.10,<2" \
  "huggingface-hub>=0.30,<2" \
  "safetensors>=0.5" \
  "pillow>=10"

"${TARGET_ENV}/bin/python" -c \
  "import torch, transformers; from transformers import AutoModelForMultimodalLM; print({'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.cuda.is_available()})"
"${TARGET_ENV}/bin/python" -m pip freeze > "logs/image_bridge_data_env_${SLURM_JOB_ID:-local}.freeze.txt"
echo "Ready: ${TARGET_ENV}"
