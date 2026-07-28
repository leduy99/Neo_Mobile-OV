#!/usr/bin/env bash
#SBATCH --job-name=neo-bptt-mem
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=output/neodragon_bptt_memory_20260728/slurm-%j.out
#SBATCH --error=output/neodragon_bptt_memory_20260728/slurm-%j.out

set -u

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
PYTHON="${CONDA_ENV}/bin/python"
OUT="${OUT:-output/neodragon_bptt_memory_20260728}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

mkdir -p "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi

run_probe() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  "${PYTHON}" tools/benchmark_neodragon_rollout_bptt.py \
    --bridge-checkpoint "${BRIDGE_CKPT}" \
    --output "${OUT}/${name}.json" \
    "$@"
}

# Each probe is a fresh process, so an OOM cannot contaminate later measurements.
run_probe window01 --window-calls 1 --optimizer-step
run_probe window03 --window-calls 3 --optimizer-step
run_probe window06 --window-calls 6 --optimizer-step
run_probe window18 --window-calls 18 --optimizer-step
run_probe window18_ckpt_all --window-calls 18 --gradient-checkpointing --checkpoint-ratio 0.0 --optimizer-step
run_probe window18_ckpt_last40 --window-calls 18 --gradient-checkpointing --checkpoint-ratio 0.6 --optimizer-step
run_probe window18_native_teacher --window-calls 18 --retain-native-text-teacher --optimizer-step
run_probe window18_native_teacher_ckpt --window-calls 18 --retain-native-text-teacher --gradient-checkpointing --checkpoint-ratio 0.0 --optimizer-step

"${PYTHON}" - "${OUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("window*.json")):
    payload = json.loads(path.read_text())
    memory = payload.get(
        "after_optimizer",
        payload.get("after_backward", payload.get("at_oom", {})),
    )
    rows.append(
        {
            "probe": path.stem,
            "status": payload.get("status"),
            "window_calls": payload.get("window_calls"),
            "checkpointing": payload.get("gradient_checkpointing"),
            "checkpoint_ratio": payload.get("checkpoint_ratio"),
            "native_text_teacher": payload.get("retain_native_text_teacher"),
            "peak_allocated_gib": memory.get("peak_allocated_gib"),
            "peak_reserved_gib": memory.get("peak_reserved_gib"),
            "forward_seconds": payload.get("forward_seconds"),
            "backward_seconds": payload.get("backward_seconds"),
            "optimizer_seconds": payload.get("optimizer_seconds"),
            "bridge_grad_norm": payload.get("bridge_grad_norm"),
        }
    )
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
