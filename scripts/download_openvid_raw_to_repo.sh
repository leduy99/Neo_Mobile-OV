#!/usr/bin/env bash
set -euo pipefail

# Download OpenVid raw video parts into this repo after captions/manifests are copied.
#
# Typical usage:
#   bash scripts/import_openvid_data_to_repo.sh
#   MAX_PARTS=1 DRY_RUN=1 bash scripts/download_openvid_raw_to_repo.sh
#   MAX_PARTS=1 bash scripts/download_openvid_raw_to_repo.sh
#
# Full run, inferred from the local recaption manifest:
#   bash scripts/download_openvid_raw_to_repo.sh
#
# Explicit part range:
#   PARTS=35-42 bash scripts/download_openvid_raw_to_repo.sh

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-download_data/data/openvid}"
MANIFEST="${MANIFEST:-}"
if [[ -z "${MANIFEST}" ]]; then
  if [[ -f "${DOWNLOAD_ROOT}/manifests/openvid_all_recaptions_merged.csv" ]]; then
    MANIFEST="${DOWNLOAD_ROOT}/manifests/openvid_all_recaptions_merged.csv"
  elif [[ -f "${DOWNLOAD_ROOT}/manifests/openvid_all_recaptions.csv" ]]; then
    MANIFEST="${DOWNLOAD_ROOT}/manifests/openvid_all_recaptions.csv"
  elif [[ -f "${DOWNLOAD_ROOT}/manifests/openvid_all.csv" ]]; then
    MANIFEST="${DOWNLOAD_ROOT}/manifests/openvid_all.csv"
  fi
fi

if [[ -z "${PARTS:-}" && ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest under ${DOWNLOAD_ROOT}/manifests." >&2
  echo "Run first: LINK_RAW=0 bash scripts/import_openvid_data_to_repo.sh" >&2
  echo "Or set PARTS=0,1,35-40 to download explicit OpenVid parts." >&2
  exit 1
fi

ARGS=(
  --download-root "${DOWNLOAD_ROOT}"
  --max-parts "${MAX_PARTS:--1}"
  --start-offset "${START_OFFSET:-0}"
  --workers "${DOWNLOAD_WORKERS:-8}"
)

if [[ -n "${PARTS:-}" ]]; then
  ARGS+=(--parts "${PARTS}")
else
  ARGS+=(--manifest "${MANIFEST}")
fi

if [[ "${EXTRACT:-1}" == "1" ]]; then
  ARGS+=(--extract)
else
  ARGS+=(--no-extract)
fi
if [[ "${OVERWRITE_EXTRACT:-0}" == "1" ]]; then
  ARGS+=(--overwrite-extract)
fi
if [[ "${KEEP_ZIP:-0}" == "1" ]]; then
  ARGS+=(--keep-zip)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python tools/data_prepare/download_openvid_raw.py "${ARGS[@]}" ${EXTRA_ARGS:-}
