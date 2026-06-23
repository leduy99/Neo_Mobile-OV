#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_mobile_ov.checkpoints import ensure_neodragon_assets, ensure_smolvlm2_checkpoint
from new_mobile_ov.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/prepare checkpoints required by New Mobile-OV.")
    parser.add_argument("--config", default="configs/mobile_ov_neodragon.yaml")
    parser.add_argument("--skip-smolvlm2", action="store_true")
    parser.add_argument("--skip-neodragon", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = load_config(args.config)
    result: dict[str, object] = {"config": args.config}

    if not args.skip_smolvlm2:
        result["smolvlm2_ckpt_path"] = ensure_smolvlm2_checkpoint(cfg.bridge.smolvlm2_ckpt_path)
    if not args.skip_neodragon and cfg.backend.name == "mobile_ov_neodragon":
        repo_path, cache_dir, model_path = ensure_neodragon_assets(
            repo_path=cfg.backend.extra.get("repo_path"),
            cache_dir=cfg.backend.extra.get("cache_dir"),
            model_id=cfg.backend.extra.get("model_id", "karnewar/Neodragon"),
            repo_url=cfg.backend.extra.get("repo_url"),
        )
        result.update(
            {
                "neodragon_repo_path": repo_path,
                "neodragon_cache_dir": cache_dir,
                "neodragon_model_path": model_path,
            }
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
