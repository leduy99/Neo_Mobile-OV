#!/usr/bin/env bash
set -euo pipefail

# CPU/I/O preprocessing. Run this before submitting GPU training jobs.
# Example:
#   bash scripts/prepare_openvid_neodragon_2s.sh

SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
  SOURCE_MANIFEST=""
fi
if [[ -z "${SOURCE_MANIFEST}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
  LOCAL_OPENVID_ROOT="${LOCAL_OPENVID_ROOT:-download_data/data/openvid}"
  if [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions_merged.csv" ]]; then
    SOURCE_MANIFEST="${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions_merged.csv"
  elif [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions.csv" ]]; then
    SOURCE_MANIFEST="${LOCAL_OPENVID_ROOT}/manifests/openvid_all_recaptions.csv"
  elif [[ -f "${LOCAL_OPENVID_ROOT}/manifests/openvid_all.csv" ]]; then
    SOURCE_MANIFEST="${LOCAL_OPENVID_ROOT}/manifests/openvid_all.csv"
  elif [[ "${ALLOW_LEGACY_SOURCE:-0}" == "1" && -f "/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv" ]]; then
    SOURCE_MANIFEST="/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv"
  elif [[ "${ALLOW_LEGACY_SOURCE:-0}" == "1" && -f "/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv" ]]; then
    SOURCE_MANIFEST="/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid/manifests/openvid_all_recaptions.csv"
  fi
fi

OUT_DIR="${OUT_DIR:-data/openvid_neodragon_2s}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_INPUT_ROWS="${MAX_INPUT_ROWS:--1}"
CLIP_POLICY="${CLIP_POLICY:-first}"
COPY_MODE="${COPY_MODE:-none}"
PROBE_MODE="${PROBE_MODE:-none}"
LOG_EVERY="${LOG_EVERY:-10000}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  python tools/data_prepare/prepare_openvid_neodragon.py \
    --source-manifest "${SOURCE_MANIFEST}" \
    --output-dir "${OUT_DIR}" \
    --max-input-rows "${MAX_INPUT_ROWS}" \
    --max-samples "${MAX_SAMPLES}" \
    --num-frames 49 \
    --target-fps 24 \
    --clip-policy "${CLIP_POLICY}" \
    --probe-mode "${PROBE_MODE}" \
    --log-every "${LOG_EVERY}" \
    --copy-mode "${COPY_MODE}"
else
  python tools/data_prepare/prepare_openvid_neodragon.py \
    --download-root "${DOWNLOAD_ROOT:-${LOCAL_OPENVID_ROOT:-download_data/data/openvid}}" \
    --download-csv \
    --download-parts "${DOWNLOAD_PARTS:-0}" \
    --start-part "${START_PART:-0}" \
    --extract-zips \
    --output-dir "${OUT_DIR}" \
    --max-input-rows "${MAX_INPUT_ROWS}" \
    --max-samples "${MAX_SAMPLES}" \
    --num-frames 49 \
    --target-fps 24 \
    --clip-policy "${CLIP_POLICY}" \
    --probe-mode "${PROBE_MODE}" \
    --log-every "${LOG_EVERY}" \
    --copy-mode "${COPY_MODE}"
fi

echo "Prepared NeoDragon OpenVid manifest: ${OUT_DIR}/manifest.csv"
