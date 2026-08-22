#!/usr/bin/env bash
# One decisive 944-prompt study: bridge/native-Qwen DreamLite anchors crossed
# with no-QuickSR/public-QuickSR, using the same Joint-v2 10K video checkpoint.
# The bridge/no-QSR cell reuses job 17351654 and is never regenerated.
#SBATCH --job-name=mov-jv2-gap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-jv2-gap-%j.out
#SBATCH --error=logs/mov-jv2-gap-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BRIDGE_ROOT="${BRIDGE_ROOT:-output/dreamlite_v11_balanced_joint_v2_step10k_vbench1x}"
NATIVE_ROOT="${NATIVE_ROOT:-output/dreamlite_native_qwen_joint_v2_step10k_vbench1x}"
DECISION_ROOT="${DECISION_ROOT:-output/joint_v2_neodragon_gap_1x}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
QSR_VARIANT="${QSR_VARIANT:-medium}"
QSR_SCALE="${QSR_SCALE:-2}"
QSR_BRIDGE_VIDEOS="${DECISION_ROOT}/videos/bridge_qsr"
QSR_NATIVE_VIDEOS="${DECISION_ROOT}/videos/native_qwen_qsr"

cd "${ROOT}"
mkdir -p logs "${DECISION_ROOT}"
test -x "${PYTHON_BIN}"
test -s "${BRIDGE_ROOT}/generation_summary.json"
test -s "${BRIDGE_ROOT}/scores/vbench/summary.json"
BRIDGE_VIDEO_COUNT=$(find "${BRIDGE_ROOT}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
if [[ "${BRIDGE_VIDEO_COUNT}" -ne 944 ]]; then
  echo "Expected 944 reusable bridge videos in ${BRIDGE_ROOT}/videos, got ${BRIDGE_VIDEO_COUNT}" >&2
  exit 1
fi

echo "Step 1/4: generate and score native-Qwen anchors with the same Joint-v2 checkpoint."
(
  export ROOT CONDA_ENV VBENCH_ENV VBENCH_REPO
  export RUN_NAME="$(basename "${NATIVE_ROOT}")"
  export RUN_ROOT="${NATIVE_ROOT}"
  export IMAGE_CONDITION=native_qwen
  export IMAGE_CONFIG=configs/mobile_ov_dreamlite_compact_v9.yaml
  export VIDEO_CKPT=checkpoints/hf_mobile_ov/mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt
  export VIDEO_HF_FILE=mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt
  export VIDEO_JOINT_CHECKPOINT=1
  export VIDEO_EXPECTED_STEP=10000
  export SAMPLES_PER_PROMPT=1
  export WITH_QUICKSR=0
  unset RECAPTION_FILE
  bash scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
)

module load buildenv-gcccuda/12.1.1-gcc12.3.0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"

echo "Step 2/4: apply public QuickSRNet to both existing video cells."
for spec in "${BRIDGE_ROOT}/videos:${QSR_BRIDGE_VIDEOS}" "${NATIVE_ROOT}/videos:${QSR_NATIVE_VIDEOS}"; do
  input_dir="${spec%%:*}"
  output_dir="${spec#*:}"
  "${PYTHON_BIN}" tools/apply_quicksrnet_video.py \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --glob '*.mp4' \
    --variant "${QSR_VARIANT}" \
    --scale "${QSR_SCALE}" \
    --checkpoint-dir checkpoints/quicksrnet \
    --device cuda \
    --dtype fp16 \
    --batch-frames 4
  count=$(find "${output_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  if [[ "${count}" -ne 944 ]]; then
    echo "QuickSR coverage failure for ${output_dir}: expected 944, got ${count}" >&2
    exit 1
  fi
done

VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
if [[ ! -f "${VBENCH_INFO}" ]]; then
  VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
fi
test -f "${VBENCH_INFO}"

echo "Step 3/4: score both QuickSR cells with the same resumable VBench evaluator."
for spec in "bridge_qsr:${QSR_BRIDGE_VIDEOS}" "native_qsr:${QSR_NATIVE_VIDEOS}"; do
  label="${spec%%:*}"
  video_dir="${spec#*:}"
  score_dir="${DECISION_ROOT}/scores/${label}"
  cd "${VBENCH_REPO}"
  "${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
    --videos "${ROOT}/${video_dir}" \
    --full-info "${VBENCH_INFO}" \
    --output "${ROOT}/${score_dir}" \
    --expected-samples-per-prompt 1
  cd "${ROOT}"
done

echo "Step 4/4: attribute the score gap."
"${PYTHON_BIN}" tools/summarize_joint_v2_neodragon_gap.py \
  --cell "bridge_no_qsr=${BRIDGE_ROOT}/scores/vbench/summary.json" \
  --cell "bridge_qsr=${DECISION_ROOT}/scores/bridge_qsr/vbench/summary.json" \
  --cell "native_no_qsr=${NATIVE_ROOT}/scores/vbench/summary.json" \
  --cell "native_qsr=${DECISION_ROOT}/scores/native_qsr/vbench/summary.json" \
  --output "${DECISION_ROOT}/decision_summary.json"

echo "Completed Joint-v2 NeoDragon gap attribution: ${DECISION_ROOT}/decision_summary.json"
