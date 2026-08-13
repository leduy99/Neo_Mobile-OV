#!/usr/bin/env bash
#SBATCH --job-name=mov-grounded-manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/mov-grounded-manifest-%j.out
#SBATCH --error=logs/mov-grounded-manifest-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
MOBILEO_ROOT="${MOBILEO_ROOT:-../Mobile-OV_Alpha/data}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${MOBILEO_ROOT}/short_caption_pretrain/manifests/short_caption_pretrain_source.csv}"
OUT_DIR="${OUT_DIR:-data/dreamlite_grounded_manifests}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${OUT_DIR}/shortcaption_verified_images.csv}"
STOP_FILE="${TMPDIR%/}/dreamlite_grounded_manifest_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -f "${SOURCE_MANIFEST}"
mkdir -p logs "${OUT_DIR}"

if [[ -s "${OUTPUT_MANIFEST}" && -s "${OUTPUT_MANIFEST}.summary.json" && "${FORCE:-0}" != "1" ]]; then
  echo "Reusing verified manifest: ${OUTPUT_MANIFEST}"
  cat "${OUTPUT_MANIFEST}.summary.json"
  exit 0
fi

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-grounded-manifest-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" tools/filter_grounded_image_manifest.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --source-name shortcaption \
  --image-path-roots "${MOBILEO_ROOT}" \
  --workers "${FILTER_WORKERS:-16}" \
  --progress-every "${PROGRESS_EVERY:-10000}"

wc -l "${OUTPUT_MANIFEST}"
cat "${OUTPUT_MANIFEST}.summary.json"
