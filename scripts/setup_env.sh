#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-neo_mobileov}"
CONDA_SH="${CONDA_SH:-/share_0/conda/etc/profile.d/conda.sh}"

source "$CONDA_SH"
conda create -y -n "$ENV_NAME" python=3.10 pip
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -e .

echo "Environment ready: $ENV_NAME"
