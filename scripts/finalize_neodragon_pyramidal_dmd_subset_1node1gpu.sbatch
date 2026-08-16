#!/usr/bin/env bash
# Index interrupted Pyramidal-DMD generation without regenerating trajectories.
# The preparation job uses atomic writes, so only complete sample_*.pt files are
# included in the partial manifest.
#SBATCH --job-name=neo-dmd-index
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/neo-dmd-index-%j.out
#SBATCH --error=logs/neo-dmd-index-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
OUT="${OUT:-data/neodragon_pyramidal_dmd_320p}"
MAX_PROMPTS="${MAX_PROMPTS:-350000}"
# Keep the interrupted-set index separate from the manifest generated after a
# future full resume, so an active DMD pilot always sees a fixed dataset.
MANIFEST_NAME="${MANIFEST_NAME:-manifest_partial.csv}"
METADATA_NAME="${METADATA_NAME:-metadata_partial.json}"

if [[ -n "${PROMPTS:-}" ]]; then
  PROMPTS="${PROMPTS}"
else
  for candidate in \
    "download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv" \
    "download_data/data/openvid/manifests/openvid_all_recaptions.csv" \
    "data/openvid_neodragon_2s/manifest.csv"; do
    if [[ -f "${candidate}" ]]; then
      PROMPTS="${candidate}"
      break
    fi
  done
fi

test -x "${PYTHON_BIN}"
test -f "${PROMPTS:-}"
test -d "${OUT}/latents"

mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"

STOP_FILE="${TMPDIR%/}/neo_dmd_index_${SLURM_JOB_ID:-local}.stop"
rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "neo-dmd-index-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

"${PYTHON_BIN}" -u tools/prepare_neodragon_dmd_synthetic_data.py \
  --config "${CONFIG}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --manifest-name "${MANIFEST_NAME}" \
  --metadata-name "${METADATA_NAME}" \
  --finalize-only

test -s "${OUT}/${MANIFEST_NAME}"
test -s "${OUT}/${METADATA_NAME}"
wc -l "${OUT}/${MANIFEST_NAME}"
cat "${OUT}/${METADATA_NAME}"
