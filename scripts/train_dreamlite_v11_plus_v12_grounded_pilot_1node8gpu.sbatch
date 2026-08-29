#!/usr/bin/env bash
# Controlled 20K continuation: V11 recipe with only the grounded source replaced by V12.
# Submit: sbatch scripts/train_dreamlite_v11_plus_v12_grounded_pilot_1node8gpu.sbatch
# Resume: OUT=output/dreamlite_v11_plus_v12_grounded_pilot/<JOBID> RESUME=auto sbatch ...
#SBATCH --job-name=mov-dream-v11v12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/mov-dream-v11v12-%j.out
#SBATCH --error=logs/mov-dream-v11v12-%j.out
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

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
TORCHRUN_BIN="${CONDA_ENV}/bin/torchrun"
CONFIG="${CONFIG:-configs/mobile_ov_dreamlite_compact_v12.yaml}"
MOBILEO_ROOT="${MOBILEO_ROOT:-../Mobile-OV_Alpha/data}"
JOURNEYDB_PROMPTS="${JOURNEYDB_PROMPTS:-${MOBILEO_ROOT}/journeydb_pretrain/manifests/journeydb_pretrain_train_ready.csv}"
SHORT_PROMPTS="${SHORT_PROMPTS:-${MOBILEO_ROOT}/short_caption_pretrain/manifests/short_caption_pretrain_source.csv}"
RELEASE_ROOT="${RELEASE_ROOT:-data/image_bridge_v1/releases/v12}"
V12_GROUNDED="${V12_GROUNDED:-${RELEASE_ROOT}/manifests/d2_grounded_high_precision_50k.csv}"
PREFLIGHT="${PREFLIGHT:-${RELEASE_ROOT}/stats/v12_data_preflight.json}"
RELEASE_SUMMARY="${RELEASE_SUMMARY:-${RELEASE_ROOT}/stats/release_summary.json}"
V11_CKPT="${V11_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
V11_HF_FILE="${V11_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
OUT="${OUT:-output/dreamlite_v11_plus_v12_grounded_pilot/${SLURM_JOB_ID:-local}}"
TARGET_STEP="${TARGET_STEP:-20000}"
RESUME="${RESUME:-none}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LR="${LR:-4e-6}"
SEMANTIC_PROBABILITY="${SEMANTIC_PROBABILITY:-0.12}"
GROUNDED_PROBABILITY="${GROUNDED_PROBABILITY:-0.10}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-512x512@1024x1024:15,768x768@1024x1024:10,1024x1024@1024x1024:15,640x400@1280x800:15,1024x640@1280x800:20,400x640@800x1280:10,640x1024@800x1280:15}"
STOP_FILE="${TMPDIR%/}/dreamlite_v11v12_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
test -s "${JOURNEYDB_PROMPTS}"
test -s "${SHORT_PROMPTS}"
test -s "${V12_GROUNDED}"
test -s "${PREFLIGHT}"
test -s "${RELEASE_SUMMARY}"

mkdir -p logs "$(dirname "${V11_CKPT}")" "${OUT}"
if [[ ! -s "${V11_CKPT}" ]]; then
  "${PYTHON_BIN}" - "${HF_REPO}" "${V11_HF_FILE}" "${V11_CKPT}" <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
import sys

repo, filename, target = sys.argv[1:]
source = Path(hf_hub_download(repo_id=repo, filename=filename))
destination = Path(target)
temporary = destination.with_suffix(destination.suffix + ".tmp")
shutil.copy2(source, temporary)
temporary.replace(destination)
print(f"Downloaded {repo}/{filename} -> {destination}")
PY
fi

# Fail before allocating model memory if either baseline or frozen V12 release is wrong.
"${PYTHON_BIN}" - "${V11_CKPT}" "${PREFLIGHT}" "${RELEASE_SUMMARY}" "${V12_GROUNDED}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

v11_path, preflight_path, release_path, grounded_path = map(Path, sys.argv[1:])
payload = torch.load(v11_path, map_location="cpu", weights_only=False)
config = payload.get("config", {})
if int(payload.get("step", -1)) != 160000:
    raise SystemExit(f"Expected V11 step 160000, got {payload.get('step')}")
if config.get("training_version") != "v11_balanced":
    raise SystemExit(f"Not the V11-balanced checkpoint: {config.get('training_version')}")
if float(config.get("semantic_prompt_probability", -1)) != 0.12:
    raise SystemExit("V11 source semantic probability is not 0.12")
if len(payload.get("bridge", {})) != 56:
    raise SystemExit(f"Unexpected V11 bridge state size: {len(payload.get('bridge', {}))}")

preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
if preflight.get("machine_preflight_passed") is not True:
    raise SystemExit(f"V12 data preflight failed: {preflight.get('machine_failures')}")
release = json.loads(release_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

matching = [
    value
    for value in release.get("manifests", {}).values()
    if Path(str(value.get("output") or value.get("path") or "")).resolve()
    == grounded_path.resolve()
]
if len(matching) != 1:
    raise SystemExit(f"Grounded manifest is not uniquely registered in V12 release: {grounded_path}")
expected = str(matching[0].get("output_sha256") or matching[0].get("sha256") or "")
if not expected or sha256(grounded_path) != expected:
    raise SystemExit(f"V12 grounded manifest hash mismatch: {grounded_path}")
print("Validated V11 initialization and immutable V12 grounded manifest")
PY

if [[ "${RESUME}" != "none" && ! -s "${OUT}/dreamlite_image_bridge_latest.pt" ]]; then
  echo "RESUME=${RESUME} requested but ${OUT}/dreamlite_image_bridge_latest.pt is missing" >&2
  exit 2
fi

"${PYTHON_BIN}" - \
  "${OUT}/run_contract.json" "${V11_CKPT}" "${JOURNEYDB_PROMPTS}" \
  "${SHORT_PROMPTS}" "${V12_GROUNDED}" "${RESUME}" "${TARGET_STEP}" \
  "${BATCH_SIZE}" "${NPROC_PER_NODE}" "${LR}" \
  "${SEMANTIC_PROBABILITY}" "${GROUNDED_PROBABILITY}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    init,
    journey,
    short,
    grounded,
    resume,
    target_step,
    batch_size,
    world_size,
    lr,
    semantic_probability,
    grounded_probability,
) = sys.argv[1:]
contract = {
    "experiment": "v11_plus_v12_grounded_pilot",
    "objective": "retain V11 curriculum while replacing only its grounded source with V12",
    "initialization": init,
    "optimizer_reset": resume == "none",
    "target_step": int(target_step),
    "global_batch": int(batch_size) * int(world_size),
    "lr": float(lr),
    "main_sources": {"journeydb": journey, "shortcaption": short},
    "main_source_weights": {"journeydb": 0.7142857143, "shortcaption": 0.2857142857},
    "semantic_prompt_probability": float(semantic_probability),
    "grounded_source": grounded,
    "grounded_batch_probability": float(grounded_probability),
    "frozen_components": ["SmolVLM2", "Qwen3-VL teacher", "DreamLite UNet"],
    "trainable_component": "5.725M-parameter DreamLite image bridge",
}
Path(output).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
print(json.dumps(contract, indent=2))
PY

PROMPT_MANIFESTS="${JOURNEYDB_PROMPTS};${SHORT_PROMPTS};${V12_GROUNDED}"
PROMPT_WEIGHTS="0.7142857143,0.2857142857,0.0"
PROMPT_NAMES="journeydb,shortcaption,grounded_cascade"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Controlled V11 + V12-grounded continuation: OUT=${OUT} RESUME=${RESUME}"
echo "steps=${TARGET_STEP} global_batch=$((BATCH_SIZE * NPROC_PER_NODE)) lr=${LR} semantic=${SEMANTIC_PROBABILITY} grounded=${GROUNDED_PROBABILITY}"
nvidia-smi || true

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "dreamlite-v11v12-${SLURM_JOB_ID:-local}" &
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

INIT_ARGS=()
if [[ "${RESUME}" == "none" ]]; then
  INIT_ARGS=(--init-bridge-checkpoint "${V11_CKPT}")
fi

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train_dreamlite_image_bridge.py \
  --config "${CONFIG}" \
  --generation-prompt-manifests "${PROMPT_MANIFESTS}" \
  --generation-source-weights "${PROMPT_WEIGHTS}" \
  --generation-source-names "${PROMPT_NAMES}" \
  --grounded-source-names grounded_cascade \
  --output-dir "${OUT}" \
  --target-step "${TARGET_STEP}" \
  --resume "${RESUME}" \
  "${INIT_ARGS[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" --lr-warmup-steps 500 --lr-final-scale 1.0 --lr-decay-end-step "${TARGET_STEP}" \
  --dtype bf16 \
  --resolution-buckets "${RESOLUTION_BUCKETS}" \
  --caption-variant-columns caption_short,caption_medium,caption_long \
  --caption-variant-weights 1,1,1 \
  --semantic-prompt-probability "${SEMANTIC_PROBABILITY}" \
  --representation-mode direct --representation-objective content_aware \
  --content-prefix-tokens 3 --content-suffix-tokens 5 \
  --content-token-mse-weight 0.5 --content-token-cos-weight 1.0 --wrapper-token-weight 0.1 \
  --content-pooled-cos-weight 0.25 --content-pooled-mse-weight 0.1 \
  --content-token-mean-weight 0.025 --content-token-std-weight 0.025 \
  --semantic-contrastive-weight 0.1 --contrastive-temperature 0.07 --global-contrastive \
  --projected-weight 1.0 --projected-content-scale 1.0 --representation-final-scale 1.0 \
  --functional-weight 5.0 --functional-cos-weight 0.1 \
  --functional-start-step 1 --functional-ramp-steps 1000 --functional-batch-size 2 \
  --functional-call-weights 1,1,1,1 \
  --grounded-functional-probability 0 --grounded-batch-probability "${GROUNDED_PROBABILITY}" \
  --grounded-functional-weight 0.5 --grounded-functional-start-step 1 \
  --transition-weight 1.0 --transition-cos-weight 0.05 \
  --student-state-probability 0.25 --student-state-start-step 1 --student-state-ramp-steps 1000 \
  --closed-loop-weight 0 \
  --training-version v11_plus_v12_grounded_pilot \
  --log-every "${LOG_EVERY:-20}" \
  --save-latest-every "${SAVE_LATEST_EVERY:-1000}" \
  --save-archive-every "${SAVE_ARCHIVE_EVERY:-5000}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -print -exec ls -lh {} \;
