#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PWD}}"
BASE_PYTHON="${BASE_PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python}"
VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
VBENCH_COMMIT="${VBENCH_COMMIT:-40e965bb183d44db976cba7d39eeb0eff85fb349}"
READY="${VBENCH_ENV}/.mobileov_vbench_ready_${VBENCH_COMMIT}"

if [[ -f "${READY}" ]] && "${VBENCH_ENV}/bin/python" -c 'import torch, vbench, detectron2' >/dev/null 2>&1; then
  echo "VBench environment already ready: ${VBENCH_ENV}"
  exit 0
fi

test -x "${BASE_PYTHON}"
if [[ ! -x "${VBENCH_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${VBENCH_ENV}"
fi
PYTHON="${VBENCH_ENV}/bin/python"
PIP="${PYTHON} -m pip"

${PIP} install --upgrade 'pip<26' 'setuptools<76' wheel ninja
if ! "${PYTHON}" -c 'import torch; assert torch.version.cuda == "12.1"' >/dev/null 2>&1; then
  ${PIP} install --upgrade \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
fi

if [[ ! -d "${VBENCH_REPO}/.git" ]]; then
  rm -rf "${VBENCH_REPO}"
  git clone https://github.com/Vchitect/VBench.git "${VBENCH_REPO}"
fi
git -C "${VBENCH_REPO}" fetch origin "${VBENCH_COMMIT}" --depth 1
git -C "${VBENCH_REPO}" checkout --detach "${VBENCH_COMMIT}"

${PIP} install -r "${VBENCH_REPO}/requirements.txt"
if ! "${PYTHON}" -c 'import detectron2' >/dev/null 2>&1; then
  # Detectron2 imports torch from setup.py, so its build must see the torch
  # already installed in the VBench environment.
  ${PIP} install --no-build-isolation \
    'detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6'
fi
${PIP} install --no-deps -e "${VBENCH_REPO}"
"${PYTHON}" -c 'import torch, vbench, detectron2; print(torch.__version__, torch.version.cuda, vbench.__file__)'
touch "${READY}"
echo "VBench environment ready: ${VBENCH_ENV}"
