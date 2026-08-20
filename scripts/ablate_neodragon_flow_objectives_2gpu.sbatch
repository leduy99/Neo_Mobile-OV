#!/usr/bin/env bash
#SBATCH --job-name=neo-pfm-ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/neo-pfm-ablate-%j.out
#SBATCH --error=logs/neo-pfm-ablate-%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
MANIFEST="${MANIFEST:-data/neodragon_flow_ablation_latents/latent_manifest.csv}"
RUN_NAME="${RUN_NAME:-flow_contract_300step_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/neodragon_flow_objective_ablation/${RUN_NAME}}"
STEPS="${STEPS:-300}"

cd "${ROOT}"
mkdir -p logs "${OUTPUT_DIR}"
test -s "${MANIFEST}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V1="checkpoints/hf_mobile_ov/mobileov_monolithic_joint_flow/mobileov_monolithic_joint_flow_17327721/neodragon_dit_bridge_latest.pt"
V2="checkpoints/hf_mobile_ov/mobileov_monolithic_joint_cfg_bridge_v2/mobileov_monolithic_joint_cfg_bridge_v2_17333147/neodragon_dit_bridge_latest.pt"
AUDIT_ARGS=()
if [[ -s "${V1}" ]]; then
  AUDIT_ARGS+=(--checkpoint "joint_v1=${V1}")
fi
if [[ -s "${V2}" ]]; then
  AUDIT_ARGS+=(--checkpoint "joint_v2=${V2}")
fi

echo "Phase 1: no-update flow-contract audit"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/audit_neodragon_flow_contract.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}/contract_audit" \
  --max-samples 4 \
  "${AUDIT_ARGS[@]}" \
  >"${OUTPUT_DIR}/contract_audit.log" 2>&1

run_cell() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" tools/ablate_neodragon_flow_training.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --cell-name "${name}" \
    --steps "${STEPS}" \
    "$@" \
    >"${OUTPUT_DIR}/${name}.log" 2>&1
}

echo "Phase 2a: isolate legacy state versus correct pyramidal endpoints"
run_cell 0 legacy_clean_data \
  --flow-contract legacy \
  --history-corrupt-max 0.0 \
  --data-weight 1.0 &
PID_A=$!
run_cell 1 pyramid_clean_data \
  --flow-contract pyramid \
  --history-corrupt-max 0.0 \
  --data-weight 1.0 &
PID_B=$!
wait "${PID_A}"
wait "${PID_B}"

echo "Phase 2b: isolate history corruption and released-teacher preservation"
run_cell 0 pyramid_corrupt_data \
  --flow-contract pyramid \
  --history-corrupt-max 0.3333333333333333 \
  --data-weight 1.0 &
PID_C=$!
run_cell 1 pyramid_corrupt_data_teacher \
  --flow-contract pyramid \
  --history-corrupt-max 0.3333333333333333 \
  --data-weight 1.0 \
  --teacher-weight 1.0 \
  --teacher-cos-weight 0.1 &
PID_D=$!
wait "${PID_C}"
wait "${PID_D}"

"${PYTHON_BIN}" tools/summarize_neodragon_flow_ablation.py \
  --run-dir "${OUTPUT_DIR}"

find "${OUTPUT_DIR}" -maxdepth 2 -type f \( -name 'report.json' -o -name 'summary.json' -o -name 'ablation_summary.json' \) -print
