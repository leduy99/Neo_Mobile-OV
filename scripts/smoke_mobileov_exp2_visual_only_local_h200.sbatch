#!/usr/bin/env bash
# Real BLIP3o/DataVerse smoke, not the full Berzelius dataset or a training job.
#SBATCH --job-name=mov-exp2-v2-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mov-exp2-v2-smoke-%j.out
#SBATCH --error=logs/mov-exp2-v2-smoke-%j.out

set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
BASE_PYTHON="${BASE_PYTHON:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
QUALITY_PYTHON="${QUALITY_PYTHON:-/share_4/users/duy/.conda/envs/vbench_mobileov_20260515/bin/python}"
CAPTION_PYTHON="${CAPTION_PYTHON:-/share_4/users/duy/.conda/envs/mobileov/bin/python}"
OUT="${EXP2_OUT:-output/exp2_visual_only_gpu_smoke/${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
"${BASE_PYTHON}" tools/data_prepare/prepare_mobileov_exp2.py --phase plan \
  --input-root "${EXP2_INPUT:-output/alignment_data_checks/stage1_live_release}" \
  --output-dir "${OUT}" --policy configs/mobileov_exp2_data.json \
  --quality-workers 1 --caption-workers 1 --limit-per-manifest "${EXP2_LIMIT:-8}"
mkdir -p "${OUT}/runtime"
SCENE_LIB="${ROOT}/${OUT}/runtime/scene_lib"
if [[ "${OUT}" = /* ]]; then SCENE_LIB="${OUT}/runtime/scene_lib"; fi
if ! PYTHONPATH="${SCENE_LIB}:${ROOT}" "${QUALITY_PYTHON}" -c 'from scenedetect.detectors import AdaptiveDetector; import importlib.metadata as m; assert m.version("scenedetect") == "0.6.7.1"' >/dev/null 2>&1; then
  "${QUALITY_PYTHON}" -m pip install --no-deps --target "${SCENE_LIB}" 'scenedetect==0.6.7.1'
fi
export PYTHONPATH="${SCENE_LIB}:${ROOT}:${PYTHONPATH:-}"
"${BASE_PYTHON}" -u tools/data_prepare/run_mobileov_exp2_data.py \
  --input-root "${EXP2_INPUT:-output/alignment_data_checks/stage1_live_release}" \
  --output-dir "${OUT}" --policy configs/mobileov_exp2_data.json \
  --quality-python "${QUALITY_PYTHON}" --caption-python "${CAPTION_PYTHON}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" --devices "${CUDA_VISIBLE_DEVICES:-0}" \
  --caption-gpus-per-worker 1 --limit-per-manifest "${EXP2_LIMIT:-8}" --max-runtime-seconds 2700
echo "Visual-only smoke complete. See ${OUT}/stage1_summary.json and ${OUT}/audit/index.html"
