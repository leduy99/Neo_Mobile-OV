#!/usr/bin/env bash
# One shared-image preparation job, then two independent 8-GPU evaluations.
set -Eeuo pipefail
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT="${PWD}"
if [[ -n "${RUN_ROOT:-}" ]]; then
  echo "Use PAIR_ROOT, not RUN_ROOT: the two evaluation jobs require separate outputs." >&2; exit 1
fi
export CHECKPOINT="${CHECKPOINT:-output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt}"
export EXPECTED_SHA256="${EXPECTED_SHA256:-a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}" MAX_PROMPTS="${MAX_PROMPTS:-0}" SEED="${SEED:-2026}"
if [[ ! "${MAX_PROMPTS}" =~ ^[0-9]+$ || ! "${SAMPLES_PER_PROMPT}" =~ ^[1-5]$ || ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Invalid prompt/sample/seed settings" >&2; exit 1
fi
if [[ "${MAX_PROMPTS}" -eq 0 && "${SAMPLES_PER_PROMPT}" -ne 5 ]]; then
  echo "Full evaluation requires five samples/prompt" >&2; exit 1
fi
test -s "${CHECKPOINT}" || { echo "Missing latest completed epoch-3 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
export PAIR_ROOT="${PAIR_ROOT:-output/vbench_stage2_epoch3_ssd1b_conditioning_${SAMPLES_PER_PROMPT}x_dual${SUFFIX}}"
export ANCHOR_ROOT="${ANCHOR_ROOT:-${PAIR_ROOT}/shared_ssd1b}"
mkdir -p logs "${PAIR_ROOT}"
exec 8>"${PAIR_ROOT}/.submit.lock"
flock -n 8 || { echo "Another submitter owns this pipeline" >&2; exit 1; }
RECEIPT="${PAIR_ROOT}/submitted_jobs.txt"
CURRENT="$(squeue --me -h -o '%i')"
if [[ -s "${RECEIPT}" ]]; then
  while IFS= read -r JOB; do
    if [[ "${JOB}" =~ ^[0-9]+$ ]] && grep -Fxq "${JOB}" <<<"${CURRENT}"; then
      echo "Pipeline already has active job ${JOB}; do not submit duplicate jobs." >&2; exit 1
    fi
  done < "${RECEIPT}"
fi
submit() {
  local ANSWER ID
  ANSWER="$(sbatch --parsable "$@")"
  ID="${ANSWER%%;*}"
  [[ "${ID}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: ${ANSWER}" >&2; return 1; }
  printf '%s\n' "${ID}"
}
PREP="$(submit scripts/prepare_vbench_stage2_ssd1b_frames_1node8gpu.sbatch)"
printf '%s\n' "${PREP}" > "${RECEIPT}"
echo "Shared SSD1B image preparation job: ${PREP}"
ANCHOR="$(submit --dependency="afterok:${PREP}" --kill-on-invalid-dep=yes scripts/vbench_stage2_epoch3_ssd1b_anchor_1node8gpu.sbatch)"
printf '%s\n' "${ANCHOR}" >> "${RECEIPT}"
echo "Text-only + SSD1B VAE anchor VBench job: ${ANCHOR}"
IMAGE="$(submit --dependency="afterok:${PREP}" --kill-on-invalid-dep=yes scripts/vbench_stage2_epoch3_image_condition_1node8gpu.sbatch)"
printf '%s\n' "${IMAGE}" >> "${RECEIPT}"
echo "Text+image conditioning, no VAE anchor VBench job: ${IMAGE}"
echo "Shared images: ${ANCHOR_ROOT}"
echo "Reports: ${PAIR_ROOT}/{ssd1b_anchor,image_condition}/quicksr_ablation.md"
