#!/usr/bin/env bash
set -euo pipefail

# Simple public entrypoint: copy the existing OpenVid caption/recaption CSVs
# into this repo. Raw videos are intentionally not copied here.

LINK_RAW=0 bash scripts/import_openvid_data_to_repo.sh "$@"
