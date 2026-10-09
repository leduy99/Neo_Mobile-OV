#!/usr/bin/env bash
# Paired end-to-end VBench: V11 + Joint V2 vs native SSD1B + released Hybrid.
# Default: export/evaluate both QuickSR and raw from the same generated frames.
#SBATCH --job-name=mov-vb-jv2nat
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-jv2nat-%j.out
#SBATCH --error=logs/mov-vb-jv2nat-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export ROOT="${PWD}"
module load buildenv-gcccuda/12.1.1-gcc12.3.0

export CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
export BASE_PYTHON="${CONDA_ENV}/bin/python"
export PYTHON_BIN="${BASE_PYTHON}"
export VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
export VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HF_HOME}}"
export TMPDIR="${TMPDIR:-${HF_HOME}}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HF_HOME}/vbench}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"

SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
EXPORT_VERSIONS="${EXPORT_VERSIONS:-both}"
case "${EXPORT_VERSIONS}" in
  both) BENCHMARK_ENTRY=tools/benchmark_joint_v2_quicksr_ablation.py ;;
  quicksr) BENCHMARK_ENTRY=tools/benchmark_joint_v2_native.py ;;
  *) echo "EXPORT_VERSIONS must be both or quicksr." >&2; exit 1 ;;
esac
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ ]]; then
  echo "MAX_PROMPTS must be nonnegative and SAMPLES_PER_PROMPT must be 1-5." >&2
  exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full benchmark requires 5 videos/prompt. Use MAX_PROMPTS for generation-only smoke tests." >&2
  exit 1
fi
read -r -a SYSTEM_ARGS <<< "${BENCHMARK_SYSTEMS:-joint_v2 native_hybrid}"
if [[ "${#SYSTEM_ARGS[@]}" -lt 1 || "${#SYSTEM_ARGS[@]}" -gt 2 ]]; then
  echo "BENCHMARK_SYSTEMS must select joint_v2, native_hybrid, or both." >&2
  exit 1
fi
for system in "${SYSTEM_ARGS[@]}"; do
  case "${system}" in
    joint_v2|native_hybrid) ;;
    *) echo "Unknown BENCHMARK_SYSTEMS entry: ${system}" >&2; exit 1 ;;
  esac
done
if [[ "${#SYSTEM_ARGS[@]}" -eq 2 && "${SYSTEM_ARGS[0]}" == "${SYSTEM_ARGS[1]}" ]]; then
  echo "Do not repeat a system in BENCHMARK_SYSTEMS." >&2
  exit 1
fi
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
if [[ "${#SYSTEM_ARGS[@]}" -eq 2 ]]; then
  DEFAULT_RUN="output/vbench_joint_v2_vs_native_quicksr_berzelius_${SAMPLES_PER_PROMPT}x${SUFFIX}"
else
  DEFAULT_RUN="output/vbench_${SYSTEM_ARGS[0]}_quicksr_berzelius_${SAMPLES_PER_PROMPT}x${SUFFIX}"
fi
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN}}"
if [[ "${EXPORT_VERSIONS}" == both && "${RUN_ROOT}" == "${DEFAULT_RUN}" ]]; then
  RUN_ROOT="${DEFAULT_RUN}_dual"
fi

JOINT_RELATIVE="mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"
if [[ -z "${JOINT_CKPT:-}" ]]; then
  # Prefer the file restored by `hf download --local-dir output`; do not duplicate it.
  if [[ -s "${ROOT}/output/${JOINT_RELATIVE}" ]]; then
    JOINT_CKPT="${ROOT}/output/${JOINT_RELATIVE}"
  else
    JOINT_CKPT="${ROOT}/checkpoints/hf_mobile_ov/${JOINT_RELATIVE}"
  fi
fi
IMAGE_CKPT="${IMAGE_CKPT:-${ROOT}/checkpoints/hf_mobile_ov/dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
NATIVE_ARGS=(--native-repo "${NATIVE_REPO:-${ROOT}/checkpoints/neodragon_repo}"
             --native-cache "${NATIVE_CACHE:-${ROOT}/checkpoints/neodragon}")
if [[ -n "${NATIVE_MODEL_DIR:-}" ]]; then
  NATIVE_ARGS+=(--native-model-dir "${NATIVE_MODEL_DIR}")
fi
REUSE_ARGS=()
for system in "${SYSTEM_ARGS[@]}"; do
  if [[ "${system}" == joint_v2 ]]; then REUSE_ROOT="${JOINT_1X:-}"; else REUSE_ROOT="${NATIVE_1X:-}"; fi
  if [[ -n "${REUSE_ROOT}" ]]; then
    if [[ "${EXPORT_VERSIONS}" == both ]]; then
      echo "Old QuickSR-only runs have no raw exports. Unset JOINT_1X/NATIVE_1X for a dual-export run." >&2
      exit 1
    fi
    test -s "${REUSE_ROOT}/generation_summary.json"
    REUSE_ARGS+=(--reuse-1x-run "${system}=${REUSE_ROOT}")
  fi
done

mkdir -p logs "${RUN_ROOT}" "${TMPDIR}" "${VBENCH_CACHE_DIR}"
test -x "${BASE_PYTHON}"
mkdir -p "$(dirname "${VBENCH_ENV}")"
# Serialize setup when two jobs share the same Python environments.
exec 9>"${VBENCH_ENV}.mobileov_setup.lock"
flock 9
bash scripts/install_dreamlite_dependencies.sh
bash scripts/install_vbench_env.sh
flock -u 9
exec 9>&-
test -x "${VBENCH_ENV}/bin/python"
test -s "${VBENCH_REPO}/vbench/VBench_full_info.json"

echo "RUN_ROOT=${RUN_ROOT}; samples/prompt=${SAMPLES_PER_PROMPT}; MAX_PROMPTS=${MAX_PROMPTS}"
echo "Selected systems: ${SYSTEM_ARGS[*]}"
echo "JOINT_CKPT=${JOINT_CKPT}; IMAGE_CKPT=${IMAGE_CKPT}"
echo "Joint: V11 image bridge + frozen Bridge V2 + monolithic DiT step 10K."
echo "Native: SSD1B native first frame + native text conditioner + released Hybrid."
echo "Both: QuickSRNet medium 2x, 49 frames, 24 FPS, 1024x640 final videos."
if [[ "${EXPORT_VERSIONS}" == both ]]; then
  echo "Also evaluate unresized 512x320 raw exports, from the exact same generated RGB frames."
  echo "Generate each video once; score both export variants separately on 8 GPUs."
fi
echo "Seed mapping matches the local 5x extension; a fresh paired run generates 9440 videos."
echo "Native and monolithic schedules differ intentionally; not an exact paper-score reproduction."
echo "If interrupted, resubmit with the SAME RUN_ROOT to resume."

srun --ntasks=1 --gres=gpu:8 "${BASE_PYTHON}" -u "${BENCHMARK_ENTRY}" \
  --output-dir "${RUN_ROOT}" \
  --systems "${SYSTEM_ARGS[@]}" \
  --protocol standard5x \
  --seed-mapping extendable \
  --joint-checkpoint "${JOINT_CKPT}" \
  --image-checkpoint "${IMAGE_CKPT}" \
  --vbench-repo "${VBENCH_REPO}" \
  --vbench-python "${VBENCH_ENV}/bin/python" \
  --workers 8 \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --seed "${SEED:-2026}" \
  "${REUSE_ARGS[@]}" \
  "${NATIVE_ARGS[@]}"

echo "Benchmark completed: ${RUN_ROOT}"
if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  if [[ "${EXPORT_VERSIONS}" == both ]]; then
    REPORT=quicksr_ablation
  elif [[ "${#SYSTEM_ARGS[@]}" -eq 2 ]]; then
    REPORT=comparison
  else
    REPORT=result
  fi
  test -s "${RUN_ROOT}/${REPORT}.json"
  test -s "${RUN_ROOT}/${REPORT}.md"
  printf '\n'
  cat "${RUN_ROOT}/${REPORT}.md"
fi
