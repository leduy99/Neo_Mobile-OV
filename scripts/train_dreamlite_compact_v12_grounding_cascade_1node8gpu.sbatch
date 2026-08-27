#!/usr/bin/env bash
# V12: V11-balanced optimization with the frozen ImageBridge-Data-v1 release.
#SBATCH --job-name=mov-dream-v12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-v12-%j.out
#SBATCH --error=logs/mov-dream-v12-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite_compact_v12.yaml}"
RELEASE_ROOT="${RELEASE_ROOT:-data/image_bridge_v1/releases/v12}"
PREFLIGHT="${PREFLIGHT:-${RELEASE_ROOT}/stats/v12_data_preflight.json}"
RELEASE_SUMMARY="${RELEASE_SUMMARY:-${RELEASE_ROOT}/stats/release_summary.json}"
MIXTURE="${MIXTURE:-${RELEASE_ROOT}/mixtures/v12_release_70_20_10.json}"
BROAD_PROMPTS="${BROAD_PROMPTS:-${RELEASE_ROOT}/manifests/d1_broad_train.csv}"
COMPOSITIONAL_PROMPTS="${COMPOSITIONAL_PROMPTS:-${RELEASE_ROOT}/manifests/d2_compositional_train.csv}"
GROUNDED_PROMPTS="${GROUNDED_PROMPTS:-${RELEASE_ROOT}/manifests/d2_grounded_high_precision_50k.csv}"
OUT="${OUT:-output/dreamlite_compact_v12_grounding_cascade/${SLURM_JOB_ID:-local}}"
TARGET_STEP="${TARGET_STEP:-160000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-none}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-512x512@1024x1024:15,768x768@1024x1024:10,1024x1024@1024x1024:15,640x400@1280x800:15,1024x640@1280x800:20,400x640@800x1280:10,640x1024@800x1280:15}"
STOP_FILE="${TMPDIR%/}/dreamlite_v12_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
test -s "${PREFLIGHT}"
test -s "${RELEASE_SUMMARY}"
test -s "${MIXTURE}"
test -s "${BROAD_PROMPTS}"
test -s "${COMPOSITIONAL_PROMPTS}"
test -s "${GROUNDED_PROMPTS}"

# Refuse to train against an unvalidated or accidentally modified release.
"${PYTHON_BIN}" - "${PREFLIGHT}" "${RELEASE_SUMMARY}" "${MIXTURE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

preflight_path = Path(sys.argv[1])
release_summary_path = Path(sys.argv[2])
mixture_path = Path(sys.argv[3])
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
release = json.loads(release_summary_path.read_text(encoding="utf-8"))
mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
if preflight.get("machine_preflight_passed") is not True:
    raise SystemExit(f"V12 machine preflight did not pass: {preflight.get('machine_failures')}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

artifacts = [*release.get("manifests", {}).values(), release.get("mixture", {})]
for artifact in artifacts:
    path = Path(str(artifact.get("output") or artifact.get("path") or ""))
    expected_hash = str(artifact.get("output_sha256") or artifact.get("sha256") or "")
    if not path.is_file() or not expected_hash or sha256(path) != expected_hash:
        raise SystemExit(f"V12 release artifact is missing or modified: {path}")
expected = {"broad": 0.7, "compositional": 0.2, "grounded": 0.1}
if mixture.get("overall_sampling") != expected:
    raise SystemExit(f"Unexpected V12 mixture: {mixture.get('overall_sampling')}")
trainer = mixture.get("trainer_contract", {})
if trainer.get("grounded_batch_probability") != 0.1:
    raise SystemExit("V12 grounded-batch probability is not 0.10")
print(
    "Validated V12 release: machine_pass=true "
    f"mixture={mixture.get('overall_sampling')} "
    f"human_audit_completed={preflight.get('human_audit', {}).get('completed_rows', 0)}"
)
PY

PROMPT_MANIFESTS="${BROAD_PROMPTS};${COMPOSITIONAL_PROMPTS};${GROUNDED_PROMPTS}"
PROMPT_WEIGHTS="0.7777777778,0.2222222222,0.0"
PROMPT_NAMES="broad,compositional,grounded_cascade"

mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Mobile-OV DreamLite V12 grounding-cascade distillation from scratch"
echo "Data mixture: broad=70% compositional=20% grounded=10%"
echo "PROMPT_MANIFESTS=${PROMPT_MANIFESTS}"
echo "OUT=${OUT} TARGET_STEP=${TARGET_STEP} RESUME=${RESUME} global_batch=$((BATCH_SIZE * NPROC_PER_NODE))"
echo "Optimization contract: V11-balanced content-aware + functional distillation"
nvidia-smi || true

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-v12-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${CONFIG}" --skip-neodragon
cleanup_heartbeat
trap - EXIT INT TERM

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config "${CONFIG}" \
  --generation-prompt-manifests "${PROMPT_MANIFESTS}" \
  --generation-source-weights "${PROMPT_WEIGHTS}" \
  --generation-source-names "${PROMPT_NAMES}" \
  --grounded-source-names grounded_cascade \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume "${RESUME}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR:-4e-5}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-2000}" \
  --lr-final-scale "${LR_FINAL_SCALE:-0.1}" \
  --lr-decay-end-step "${LR_DECAY_END_STEP:-120000}" \
  --dtype bf16 \
  --resolution-buckets "${RESOLUTION_BUCKETS}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --semantic-prompt-probability 0 \
  --representation-mode direct \
  --representation-objective content_aware \
  --content-prefix-tokens 3 \
  --content-suffix-tokens 5 \
  --content-token-mse-weight 0.5 \
  --content-token-cos-weight 1.0 \
  --wrapper-token-weight 0.1 \
  --content-pooled-cos-weight 0.25 \
  --content-pooled-mse-weight 0.1 \
  --content-token-mean-weight 0.025 \
  --content-token-std-weight 0.025 \
  --semantic-contrastive-weight 0.1 \
  --contrastive-temperature 0.07 \
  --global-contrastive \
  --projected-weight 1.0 \
  --projected-content-scale 1.0 \
  --representation-final-scale 1.0 \
  --functional-weight 5.0 \
  --functional-cos-weight 0.1 \
  --functional-start-step 10001 \
  --functional-ramp-steps 20000 \
  --functional-batch-size "${FUNCTIONAL_BATCH_SIZE:-2}" \
  --functional-call-weights 1,1,1,1 \
  --grounded-functional-probability 0 \
  --grounded-batch-probability 0.10 \
  --grounded-functional-weight 0.5 \
  --transition-weight 1.0 \
  --transition-cos-weight 0.05 \
  --student-state-probability 0.25 \
  --student-state-start-step 30001 \
  --student-state-ramp-steps 20000 \
  --closed-loop-weight 0 \
  --training-version v12_grounding_cascade \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-20000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
