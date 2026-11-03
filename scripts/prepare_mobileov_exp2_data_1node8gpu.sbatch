#!/usr/bin/env bash
# Separate Exp-2 views; the old release and raw TAR archives are read-only.
#SBATCH --job-name=mov-exp2-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-exp2-data-%j.out
#SBATCH --error=logs/mov-exp2-data-%j.out

set -Eeuo pipefail
module load buildenv-nvhpc/24.5-cuda12.4
ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "${ROOT}"
BASE_PYTHON="${BASE_PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python}"
QUALITY_PYTHON="${QUALITY_PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_vbench/bin/python}"
CAPTION_PYTHON="${CAPTION_PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_image_data/bin/python}"
export EXP2_OUT="${EXP2_OUT:-download_data/data/mobileov_exp2_visual_only_v2}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HF_HOME}/pip_exp2}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
mkdir -p logs "${PIP_CACHE_DIR}" "${TMPDIR}"
# Validate separation before creating anything inside the requested data output.
"${BASE_PYTHON}" tools/data_prepare/prepare_mobileov_exp2.py --phase plan \
  --input-root "${EXP2_INPUT:-download_data/data/univideo_stage1_expanded650}" \
  --output-dir "${EXP2_OUT}" --policy "${EXP2_POLICY:-configs/mobileov_exp2_data.json}" \
  --quality-workers 8 --caption-workers 4 --limit-per-manifest "${EXP2_LIMIT:-0}"
mkdir -p "${EXP2_OUT}/runtime"

# A small isolated dependency, not an upgrade of either existing environment.
exec 9>"${EXP2_OUT}/runtime/setup.lock"
flock 9
SCENE_LIB="${ROOT}/${EXP2_OUT}/runtime/scene_lib"
if [[ "${EXP2_OUT}" = /* ]]; then SCENE_LIB="${EXP2_OUT}/runtime/scene_lib"; fi
if ! PYTHONPATH="${SCENE_LIB}:${ROOT}" "${QUALITY_PYTHON}" -c 'from scenedetect.detectors import AdaptiveDetector; import importlib.metadata as m; assert m.version("scenedetect") == "0.6.7.1"' >/dev/null 2>&1; then
  "${QUALITY_PYTHON}" -m pip install --no-deps --upgrade --target "${SCENE_LIB}" 'scenedetect==0.6.7.1'
fi
flock -u 9
export PYTHONPATH="${SCENE_LIB}:${ROOT}:${PYTHONPATH:-}"
"${QUALITY_PYTHON}" -c 'import av, cv2, clip, pyiqa; from scenedetect.detectors import AdaptiveDetector; print("Quality dependencies ready")'
"${CAPTION_PYTHON}" -c 'from transformers import AutoModelForMultimodalLM; print("Caption loader ready")'

set +e
srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u tools/data_prepare/run_mobileov_exp2_data.py \
  --input-root "${EXP2_INPUT:-download_data/data/univideo_stage1_expanded650}" \
  --output-dir "${EXP2_OUT}" --policy "${EXP2_POLICY:-configs/mobileov_exp2_data.json}" \
  --quality-python "${QUALITY_PYTHON}" --caption-python "${CAPTION_PYTHON}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" --vbench-repo "${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}" \
  --devices "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" --phase "${EXP2_PHASE:-all}" \
  --limit-per-manifest "${EXP2_LIMIT:-0}" --max-runtime-seconds "${MAX_RUNTIME_SECONDS:-237600}"
rc=$?
set -e
if [[ "${rc}" == 75 && "${AUTO_CONTINUE:-1}" == 1 ]]; then
  sbatch --dependency="afterany:${SLURM_JOB_ID}" --export=ALL "${ROOT}/scripts/prepare_mobileov_exp2_data_1node8gpu.sbatch"
  echo "Queued data continuation; same policy, output, and per-sample receipts."
  exit 0
fi
exit "${rc}"
