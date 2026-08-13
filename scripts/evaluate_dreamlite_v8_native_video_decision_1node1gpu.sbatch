#!/usr/bin/env bash
# Final evaluation-only factorial ablation. It reuses the two completed 20-prompt
# runs, scores four fixed video branches, and writes contact sheets plus paired effects.
# Submit: sbatch scripts/evaluate_dreamlite_v8_native_video_decision_1node1gpu.sbatch
# No image/video generation occurs in this job.
#SBATCH --job-name=mov-v8decision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-v8decision-%j.out
#SBATCH --error=logs/mov-v8decision-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-gcccuda/12.1.1-gcc12.3.0

ROOT="${ROOT:-$(pwd -P)}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
OUT="${OUT:-output/dreamlite_v8_native_video_decision}"
RUN_ROOTS_CSV="${RUN_ROOTS_CSV:-output/dreamlite_v8_native_video_control20/17266835,output/dreamlite_v8_native_video_control20/17266836}"
STOP_FILE="${TMPDIR:-/tmp}/mov_v8_decision_${SLURM_JOB_ID:-local}.stop"

cd "${ROOT}"
OUT="$(realpath -m "${OUT}")"
mkdir -p logs "${OUT}"
test -x "${PYTHON_BIN}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export XDG_CACHE_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export VBENCH_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches/vbench
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-v8-decision-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

export ROOT BASE_PYTHON="${PYTHON_BIN}" VBENCH_ENV VBENCH_REPO
bash scripts/install_vbench_env.sh

VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
[[ -f "${VBENCH_INFO}" ]] || VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
test -f "${VBENCH_INFO}"

IFS=',' read -r -a RUN_ROOTS <<< "${RUN_ROOTS_CSV}"
PREPARE_ARGS=(--run-roots)
for run_root in "${RUN_ROOTS[@]}"; do
  run_root="${run_root//[[:space:]]/}"
  test -f "${run_root}/summary.json"
  PREPARE_ARGS+=("${run_root}")
done
PREPARE_ARGS+=(--vbench-info "${VBENCH_INFO}" --output "${OUT}")

echo "OUT=${OUT}"
echo "RUN_ROOTS=${RUN_ROOTS_CSV}"
nvidia-smi || true
"${PYTHON_BIN}" -u tools/prepare_dreamlite_v8_decision_ablation.py "${PREPARE_ARGS[@]}"

SEEDS=$("${PYTHON_BIN}" - "${OUT}/prepared.json" <<'PY'
import json, sys
print(','.join(str(value) for value in json.load(open(sys.argv[1]))['seeds']))
PY
)
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
mapfile -t DIMENSION_SPECS < <("${PYTHON_BIN}" - "${OUT}/prepared.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
all_dimensions = [
    'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree',
    'aesthetic_quality', 'imaging_quality', 'temporal_flickering', 'object_class',
    'multiple_objects', 'color', 'spatial_relationship', 'scene', 'temporal_style',
    'overall_consistency', 'human_action', 'appearance_style',
]
for dimension in payload['active_dimensions']:
    print(f"{all_dimensions.index(dimension)}:{dimension}")
PY
)
(( ${#DIMENSION_SPECS[@]} > 0 ))

VIDEO_CELLS=(
  image_native_qwen__text_native_neodragon
  image_native_qwen__text_exp1_64k
  image_v8_imageonly__text_native_neodragon
  image_v8_imageonly__text_exp1_64k
)
for seed in "${SEED_LIST[@]}"; do
  for cell in "${VIDEO_CELLS[@]}"; do
    echo "Scoring generated video seed=${seed} cell=${cell}"
    for dimension_spec in "${DIMENSION_SPECS[@]}"; do
      dimension_index="${dimension_spec%%:*}"
      dimension="${dimension_spec#*:}"
      echo "Scoring dimension=${dimension} seed=${seed} cell=${cell}"
      (
        cd "${VBENCH_REPO}"
        "${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
          --videos "${OUT}/inputs/seed_${seed}/branches/${cell}/videos" \
          --full-info "${OUT}/vbench_dimension_info/${dimension}.json" \
          --output "${OUT}/scores/seed_${seed}/${cell}" \
          --start "${dimension_index}" --end "$((dimension_index + 1))" \
          --expected-samples-per-prompt 1
      )
    done
  done
done

"${PYTHON_BIN}" -u tools/summarize_dreamlite_v8_decision_ablation.py --root "${OUT}"

cleanup_heartbeat
trap - EXIT INT TERM
echo "Completed final DreamLite V8 / NeoDragon decision ablation."
cat "${OUT}/decision_summary.json"
