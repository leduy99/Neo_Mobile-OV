#!/usr/bin/env bash
#SBATCH --job-name=mov-imgdata-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/mov-imgdata-v1-%j.out
#SBATCH --error=logs/mov-imgdata-v1-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
MOBILEO_ROOT="${MOBILEO_ROOT:-../Mobile-OV_Alpha/data}"
JOURNEYDB_MANIFEST="${JOURNEYDB_MANIFEST:-${MOBILEO_ROOT}/journeydb_pretrain/manifests/journeydb_pretrain_train_ready.csv}"
SHORT_MANIFEST="${SHORT_MANIFEST:-${MOBILEO_ROOT}/short_caption_pretrain/manifests/short_caption_pretrain_source.csv}"
GROUNDED_MANIFEST="${GROUNDED_MANIFEST:-data/dreamlite_grounded_manifests/shortcaption_verified_images.csv}"
COMPOSITION_MANIFESTS="${COMPOSITION_MANIFESTS:-}"
COMPOSITION_NAMES="${COMPOSITION_NAMES:-}"
OUT_DIR="${OUT_DIR:-data/image_bridge_v1}"
STOP_FILE="${TMPDIR%/}/image_bridge_data_v1_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -s "${JOURNEYDB_MANIFEST}"
test -s "${SHORT_MANIFEST}"
test -s "${GROUNDED_MANIFEST}"
mkdir -p logs "$(dirname "${OUT_DIR}")" "${TMPDIR}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "image-bridge-data-v1-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ARGS=(
  --broad-manifests "${JOURNEYDB_MANIFEST};${SHORT_MANIFEST}"
  --broad-source-names "journeydb,shortcaption"
  --grounded-manifests "${GROUNDED_MANIFEST}"
  --grounded-source-names "shortcaption_grounded"
  --output-dir "${OUT_DIR}"
  --max-broad-records "${MAX_BROAD_RECORDS:-3000000}"
  --max-compositional-records "${MAX_COMPOSITIONAL_RECORDS:-200000}"
  --max-grounded-records "${MAX_GROUNDED_RECORDS:-200000}"
  --validation-fraction "${VALIDATION_FRACTION:-0.005}"
  --hard-validation-per-capability "${HARD_VALIDATION_PER_CAPABILITY:-100}"
  --progress-every "${PROGRESS_EVERY:-100000}"
)

if [[ -n "${COMPOSITION_MANIFESTS}" ]]; then
  ARGS+=(--compositional-manifests "${COMPOSITION_MANIFESTS}")
  if [[ -n "${COMPOSITION_NAMES}" ]]; then
    ARGS+=(--compositional-source-names "${COMPOSITION_NAMES}")
  fi
fi
if [[ "${SKIP_SOURCE_HASHES:-0}" == "1" ]]; then
  ARGS+=(--skip-source-hashes)
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

"${PYTHON_BIN}" tools/build_image_bridge_data_v1.py "${ARGS[@]}"

cat "${OUT_DIR}/stats/summary.json"
cat "${OUT_DIR}/mixtures/v12_70_20_10.json"
