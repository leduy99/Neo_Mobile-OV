#!/usr/bin/env bash
# One full OpenVid pass of DistillT5-style, all-call DreamLite response KD.
# Submit: sbatch scripts/train_dreamlite_openvid_step_following_full_1node8gpu.sbatch
# Resume the same output after interruption:
# OUT=output/dreamlite_openvid_step_following_full/<OLD_JOB_ID> RESUME=auto sbatch ...
#SBATCH --job-name=mov-dream-stepkd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-stepkd-%j.out
#SBATCH --error=logs/mov-dream-stepkd-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
PROMPTS="${PROMPTS:-download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv}"
V11_CKPT="${V11_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
V11_HF_FILE="${V11_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
OUT="${OUT:-output/dreamlite_openvid_step_following_full/${SLURM_JOB_ID:-local}}"
RESUME="${RESUME:-none}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-512x512@1024x1024:15,768x768@1024x1024:10,1024x1024@1024x1024:15,640x400@1280x800:15,1024x640@1280x800:20,400x640@800x1280:10,640x1024@800x1280:15}"
STOP_FILE="${TMPDIR%/}/dreamlite_stepkd_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
test -s "${PROMPTS}"
if (( BATCH_SIZE < 1 || NPROC_PER_NODE < 1 )); then
  echo "BATCH_SIZE and NPROC_PER_NODE must be positive." >&2
  exit 2
fi

PROMPT_ROWS=$(( $(wc -l < "${PROMPTS}") - 1 ))
if (( PROMPT_ROWS < 1 )); then
  echo "Prompt manifest is empty: ${PROMPTS}" >&2
  exit 2
fi
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))
# One optimizer step consumes one trajectory per sample and supervises all four calls.
FULL_PASS_STEPS=$(( PROMPT_ROWS / GLOBAL_BATCH ))
TARGET_STEP="${TARGET_STEP:-${FULL_PASS_STEPS}}"
if (( TARGET_STEP < 1 )); then
  echo "TARGET_STEP must be positive." >&2
  exit 2
fi

mkdir -p logs "$(dirname "${V11_CKPT}")" "${OUT}"
if [[ ! -s "${V11_CKPT}" ]]; then
  "${PYTHON_BIN}" - "${HF_REPO}" "${V11_HF_FILE}" "${V11_CKPT}" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

repo, filename, target = sys.argv[1:]
source = Path(hf_hub_download(repo_id=repo, filename=filename))
destination = Path(target)
temporary = destination.with_suffix(destination.suffix + ".tmp")
shutil.copy2(source, temporary)
temporary.replace(destination)
print(f"Downloaded {repo}/{filename} -> {destination}")
PY
fi

# Validate both the baseline identity and the text-only data contract before
# loading Qwen3-VL or DreamLite onto eight GPUs.
"${PYTHON_BIN}" - "${V11_CKPT}" "${PROMPTS}" "${OUT}/run_contract.json" \
  "${PROMPT_ROWS}" "${GLOBAL_BATCH}" "${TARGET_STEP}" "${LR}" <<'PY'
import csv
import json
from pathlib import Path
import sys

import torch

checkpoint_path = Path(sys.argv[1])
prompt_path = Path(sys.argv[2])
contract_path = Path(sys.argv[3])
prompt_rows, global_batch, target_step = map(int, sys.argv[4:7])
lr = float(sys.argv[7])

payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = payload.get("config", {})
if int(payload.get("step", -1)) != 160000:
    raise SystemExit(f"Expected V11-balanced step 160000, got {payload.get('step')}")
if config.get("training_version") != "v11_balanced":
    raise SystemExit(f"Expected V11-balanced, got {config.get('training_version')}")
state = payload.get("bridge")
if not isinstance(state, dict) or len(state) != 56:
    raise SystemExit(f"Unexpected V11 bridge state: {type(state).__name__}, keys={len(state or {})}")

with prompt_path.open("r", encoding="utf-8", newline="") as handle:
    columns = set(next(csv.reader(handle)))
required = {"caption_short", "caption_medium", "caption_long"}
missing = sorted(required - columns)
if missing:
    raise SystemExit(f"OpenVid caption columns missing: {missing}")

contract = {
    "recipe": "dreamlite_openvid_step_following_full_v1",
    "initialization": str(checkpoint_path),
    "initialization_step": 160000,
    "prompt_manifest": str(prompt_path),
    "prompt_rows": prompt_rows,
    "caption_variants": ["caption_short", "caption_medium", "caption_long"],
    "caption_variant_sampling": "equal probability per row",
    "global_batch": global_batch,
    "target_steps": target_step,
    "approximate_dataset_passes": target_step * global_batch / prompt_rows,
    "dreamlite_calls_per_sample": 4,
    "optimized_objective": "mean plain MSE over all frozen-DreamLite predictions",
    "state_policy": "native Qwen3-VL teacher prediction advances every state",
    "trainable": "DreamLite image bridge only; SmolVLM2, Qwen3-VL, and DreamLite are frozen",
    "representation_loss": False,
    "grounded_image_loss": False,
    "synthetic_hard_prompt_mix": False,
    "learning_rate": lr,
}
contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
print(json.dumps(contract, indent=2))
PY

export PATH="${CONDA_ENV}/bin:${PATH}"
rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-stepkd-setup-${SLURM_JOB_ID:-local}" &
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

if [[ "${RESUME}" == "auto" && -s "${OUT}/dreamlite_image_bridge_latest.pt" ]]; then
  CHECKPOINT_ARGS=(--resume auto)
else
  CHECKPOINT_ARGS=(--resume none --init-bridge-checkpoint "${V11_CKPT}")
fi

echo "DreamLite OpenVid full step-following vision KD"
echo "rows=${PROMPT_ROWS} global_batch=${GLOBAL_BATCH} target_steps=${TARGET_STEP} all_call_examples=$((TARGET_STEP * GLOBAL_BATCH * 4))"
echo "OUT=${OUT} INIT=${V11_CKPT} LR=${LR}"
nvidia-smi || true

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config "${CONFIG}" \
  --generation-prompts "${PROMPTS}" \
  --generation-source-names openvid \
  --generation-source-weights 1 \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  "${CHECKPOINT_ARGS[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-1000}" \
  --lr-final-scale "${LR_FINAL_SCALE:-0.1}" \
  --lr-decay-end-step "${TARGET_STEP}" \
  --dtype bf16 \
  --resolution-buckets "${RESOLUTION_BUCKETS}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --semantic-prompt-probability 0 \
  --representation-objective direct \
  --skip-representation-loss \
  --functional-objective teacher_following \
  --functional-weight 1 \
  --functional-cos-weight 0 \
  --functional-start-step 1 \
  --functional-ramp-steps 0 \
  --functional-batch-size "${BATCH_SIZE}" \
  --grounded-functional-probability 0 \
  --grounded-batch-probability 0 \
  --transition-weight 0 \
  --transition-cos-weight 0 \
  --student-state-probability 0 \
  --closed-loop-weight 0 \
  --training-version dreamlite_openvid_step_following_full_v1 \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-5000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-20000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
