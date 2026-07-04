#!/usr/bin/env bash
set -euo pipefail

# Bring the OpenVid training data contract into this repo.
#
# Default behavior is safe for huge datasets:
#   - manifest/recaption CSVs are copied into ./download_data/data/openvid/manifests
#   - recaption part CSVs are symlinked/copied into ./download_data/data/openvid/recaption
#   - raw videos are symlinked unless LINK_RAW=0
#
# Examples:
#   bash scripts/import_openvid_data_to_repo.sh
#   LINK_RAW=1 bash scripts/import_openvid_data_to_repo.sh
#   LINK_MODE=hardlink bash scripts/import_openvid_data_to_repo.sh
#   SRC_ROOT=/path/to/openvid DEST_ROOT=download_data/data/openvid bash scripts/import_openvid_data_to_repo.sh

SRC_ROOT="${SRC_ROOT:-/proj/cvl/users/x_fahkh2/Mobile-OV_Alpha/download_data/data/openvid}"
SRC_MANIFEST_DIR="${SRC_MANIFEST_DIR:-${SRC_ROOT}/manifests}"
SRC_RECAPTION_DIR="${SRC_RECAPTION_DIR:-${SRC_ROOT}/recaption}"
SRC_REWRITE_PREFIX="${SRC_REWRITE_PREFIX:-${SRC_ROOT}}"
DEST_ROOT="${DEST_ROOT:-download_data/data/openvid}"
DEST_MANIFEST_DIR="${DEST_ROOT}/manifests"
DEST_RECAPTION_DIR="${DEST_ROOT}/recaption"
LINK_MODE="${LINK_MODE:-symlink}"  # symlink | hardlink | copy
LINK_RAW="${LINK_RAW:-0}"

if [[ ! -d "${SRC_MANIFEST_DIR}" ]]; then
  echo "Missing SRC_MANIFEST_DIR=${SRC_MANIFEST_DIR}" >&2
  echo "Set SRC_MANIFEST_DIR=/path/to/openvid/manifests, or use FORCE_DOWNLOAD=1 bash scripts/prepare_openvid_neodragon_2s.sh" >&2
  exit 1
fi

mkdir -p "${DEST_MANIFEST_DIR}" "${DEST_RECAPTION_DIR}"

SRC_ABS="$(cd "${SRC_ROOT}" 2>/dev/null && pwd || printf '%s' "${SRC_REWRITE_PREFIX}")"
DEST_ABS="$(mkdir -p "${DEST_ROOT}" && cd "${DEST_ROOT}" && pwd)"

copy_plain_file() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] || return 0
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
  echo "copied ${src} -> ${dst}"
}

copy_manifest_csv() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] || return 0
  mkdir -p "$(dirname "${dst}")"
  python - "$src" "$dst" "$SRC_ABS" "$DEST_ABS" <<'PY'
import csv
import sys
from pathlib import Path

src, dst, src_abs, dest_abs = sys.argv[1:5]
with open(src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = []
    fieldnames = reader.fieldnames or []
    for row in reader:
        for key, value in list(row.items()):
            if isinstance(value, str) and src_abs in value:
                row[key] = value.replace(src_abs, dest_abs)
        rows.append(row)
Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
  echo "copied+rewrote ${src} -> ${dst}"
}

link_tree() {
  local src="$1"
  local dst="$2"
  [[ -e "${src}" ]] || return 0
  rm -rf "${dst}"
  mkdir -p "$(dirname "${dst}")"
  case "${LINK_MODE}" in
    symlink)
      ln -s "$(cd "$(dirname "${src}")" && pwd)/$(basename "${src}")" "${dst}"
      ;;
    hardlink)
      cp -al "${src}" "${dst}"
      ;;
    copy)
      cp -a "${src}" "${dst}"
      ;;
    *)
      echo "Unsupported LINK_MODE=${LINK_MODE}; use symlink, hardlink, or copy." >&2
      exit 1
      ;;
  esac
  echo "${LINK_MODE} ${src} -> ${dst}"
}

copy_manifest_csv "${SRC_MANIFEST_DIR}/openvid_all.csv" "${DEST_MANIFEST_DIR}/openvid_all.csv"
copy_plain_file "${SRC_MANIFEST_DIR}/openvid_all.summary.json" "${DEST_MANIFEST_DIR}/openvid_all.summary.json"
copy_manifest_csv "${SRC_MANIFEST_DIR}/openvid_all_recaptions.csv" "${DEST_MANIFEST_DIR}/openvid_all_recaptions.csv"
copy_manifest_csv "${SRC_MANIFEST_DIR}/openvid_all_recaptions_merged.csv" "${DEST_MANIFEST_DIR}/openvid_all_recaptions_merged.csv"

if [[ -d "${SRC_RECAPTION_DIR}" ]]; then
  for entry in "${SRC_RECAPTION_DIR}"/*; do
    [[ -e "${entry}" ]] || continue
    link_tree "${entry}" "${DEST_RECAPTION_DIR}/$(basename "${entry}")"
  done
fi

if [[ "${LINK_RAW}" == "1" && -d "${SRC_ROOT}/raw" ]]; then
  link_tree "${SRC_ROOT}/raw" "${DEST_ROOT}/raw"
fi

echo
echo "OpenVid data is now available under: ${DEST_ROOT}"
echo "Preferred recaption manifest:"
if [[ -f "${DEST_ROOT}/manifests/openvid_all_recaptions_merged.csv" ]]; then
  echo "  ${DEST_ROOT}/manifests/openvid_all_recaptions_merged.csv"
elif [[ -f "${DEST_ROOT}/manifests/openvid_all_recaptions.csv" ]]; then
  echo "  ${DEST_ROOT}/manifests/openvid_all_recaptions.csv"
elif [[ -f "${DEST_ROOT}/manifests/openvid_all.csv" ]]; then
  echo "  ${DEST_ROOT}/manifests/openvid_all.csv"
else
  echo "  not found; run recaption merge or download OpenVid first"
fi
