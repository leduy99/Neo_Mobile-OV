#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${CONDA_ENV}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade \
  "transformers==4.57.3" \
  "diffusers==0.39.0" \
  "qwen-vl-utils==0.0.14"

"${PYTHON_BIN}" - <<'PY'
import diffusers
import transformers
from diffusers import DreamLiteMobilePipeline, DreamLiteUNetModel
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

print(f"DreamLite environment ready: diffusers={diffusers.__version__} transformers={transformers.__version__}")
PY
