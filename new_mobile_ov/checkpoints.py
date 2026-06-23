from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)

DEFAULT_SMOLVLM2_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_NEODRAGON_MODEL_ID = "karnewar/Neodragon"
DEFAULT_NEODRAGON_REPO_URL = "https://github.com/Qualcomm-AI-research/neodragon.git"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | os.PathLike[str] | None, *, default: str | None = None) -> Path:
    raw = str(path or default or "").strip()
    if not raw:
        raise ValueError("A checkpoint path is required")
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    return repo_root() / expanded


def _rank_label() -> str:
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank is not None:
        return f"rank={rank} local_rank={local_rank or '?'}"
    return f"pid={os.getpid()}"


def _with_file_lock(target: Path, action, *, timeout_s: float | None = None) -> None:
    """Run an action once while other ranks wait for target to appear."""

    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = float(timeout_s or os.environ.get("MOBILEOV_CHECKPOINT_LOCK_TIMEOUT", 7200))
    lock_path = target.with_name(target.name + ".lock")
    start = time.time()
    acquired = False
    fd = None
    while not acquired:
        if target.exists():
            return
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{_rank_label()} host={os.uname().nodename}\n".encode("utf-8"))
            acquired = True
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for checkpoint lock: {lock_path}")
            time.sleep(5.0)

    try:
        if target.exists():
            return
        action()
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _copy_checkpoint(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint source does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def _download_hf_file(repo_id: str, filename: str, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    _copy_checkpoint(downloaded, target)


def _download_or_copy_smolvlm2_checkpoint(target: Path) -> bool:
    """Return True when a user-provided converted checkpoint was copied/downloaded."""

    source = os.environ.get("MOBILEOV_SMOLVLM2_CKPT_SOURCE", "").strip()
    hf_repo = os.environ.get("MOBILEOV_SMOLVLM2_CKPT_HF_REPO", "").strip()
    hf_file = os.environ.get("MOBILEOV_SMOLVLM2_CKPT_HF_FILE", "").strip()

    if source:
        LOGGER.info("Preparing SmolVLM2 checkpoint from MOBILEOV_SMOLVLM2_CKPT_SOURCE=%s", source)
        if source.startswith("hf://"):
            rel = source[len("hf://") :]
            parts = rel.split("/")
            if len(parts) < 3:
                raise ValueError(f"Expected hf://org/repo/path/to/file.pt, got {source}")
            _download_hf_file("/".join(parts[:2]), "/".join(parts[2:]), target)
            return True
        if source.startswith("file://"):
            source = source[len("file://") :]
        _copy_checkpoint(Path(source).expanduser(), target)
        return True

    if hf_repo and hf_file:
        LOGGER.info("Preparing SmolVLM2 checkpoint from HF repo=%s file=%s", hf_repo, hf_file)
        _download_hf_file(hf_repo, hf_file, target)
        return True

    return False


def _convert_smolvlm2_from_hf(model_id: str, target: Path) -> None:
    try:
        from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "SmolVLM2 checkpoint is missing and transformers is required to convert it from Hugging Face. "
            "Install transformers or set MOBILEOV_SMOLVLM2_CKPT_SOURCE to a converted .pt file."
        ) from exc

    device_name = os.environ.get("MOBILEOV_SMOLVLM2_CONVERT_DEVICE")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    LOGGER.info("Converting SmolVLM2 from HF model_id=%s to %s on %s", model_id, target, device)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except Exception as exc:
        LOGGER.warning("AutoModelForImageTextToText failed (%s); falling back to AutoModel.", exc)
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()

    config = getattr(model, "config", None)
    config_dict = None
    if config is not None:
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = dict(config.__dict__)

    tokenizer_vocab = None
    tokenizer_config = None
    if tokenizer is not None:
        if hasattr(tokenizer, "get_vocab"):
            tokenizer_vocab = tokenizer.get_vocab()
        if hasattr(tokenizer, "init_kwargs"):
            tokenizer_config = tokenizer.init_kwargs

    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config_dict": config_dict,
        "config": config,
        "tokenizer_vocab": tokenizer_vocab,
        "tokenizer_config": tokenizer_config,
        "model": model,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "checkpoint_format": "state_dict",
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, target)
    LOGGER.info("Saved converted SmolVLM2 checkpoint: %s", target)


def ensure_smolvlm2_checkpoint(path: str | os.PathLike[str] | None = None) -> str:
    """Ensure a converted SmolVLM2 `.pt` checkpoint exists inside this repo.

    Precedence when the target is missing:
    1. copy/download `MOBILEOV_SMOLVLM2_CKPT_SOURCE`
    2. download `MOBILEOV_SMOLVLM2_CKPT_HF_REPO` + `MOBILEOV_SMOLVLM2_CKPT_HF_FILE`
    3. convert `MOBILEOV_SMOLVLM2_MODEL_ID` from raw Hugging Face weights
    """

    target = resolve_repo_path(path, default="checkpoints/smolvlm2_500m/smolvlm2_500m.pt")
    if target.exists():
        return str(target)
    if os.environ.get("MOBILEOV_SMOLVLM2_AUTO_DOWNLOAD", "1").strip().lower() in {"0", "false", "no", "off"}:
        raise FileNotFoundError(
            f"Missing SmolVLM2 checkpoint: {target}. "
            "Enable auto-download or set MOBILEOV_SMOLVLM2_CKPT_SOURCE."
        )

    def action() -> None:
        if _download_or_copy_smolvlm2_checkpoint(target):
            return
        model_id = os.environ.get("MOBILEOV_SMOLVLM2_MODEL_ID", DEFAULT_SMOLVLM2_MODEL_ID).strip()
        _convert_smolvlm2_from_hf(model_id, target)

    _with_file_lock(target, action)
    return str(target)


def _clone_repo(repo_url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.{os.getpid()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(tmp)], check=True)
    os.replace(tmp, target)


def ensure_neodragon_assets(
    *,
    repo_path: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    model_id: str | None = None,
    repo_url: str | None = None,
) -> tuple[str, str, str]:
    """Ensure Neodragon source and Hugging Face weights are locally available."""

    repo = resolve_repo_path(repo_path, default="checkpoints/neodragon_repo")
    cache = resolve_repo_path(cache_dir, default="checkpoints/neodragon")
    model_id = (model_id or DEFAULT_NEODRAGON_MODEL_ID).strip()
    repo_url = (repo_url or os.environ.get("MOBILEOV_NEODRAGON_REPO_URL") or DEFAULT_NEODRAGON_REPO_URL).strip()

    if not (repo / "neodragon").exists():
        LOGGER.info("Neodragon repo missing; cloning %s into %s", repo_url, repo)
        clone_marker = repo.parent / f".{repo.name}.clone_ready"

        def clone_action() -> None:
            if not (repo / "neodragon").exists():
                _clone_repo(repo_url, repo)
            clone_marker.touch()

        _with_file_lock(clone_marker, clone_action)
    if not (repo / "neodragon").exists():
        raise FileNotFoundError(f"Neodragon clone did not produce expected package directory: {repo / 'neodragon'}")

    cache.mkdir(parents=True, exist_ok=True)
    model_cache = cache / f"models--{model_id.replace('/', '--')}"
    model_marker = cache / f".{model_id.replace('/', '--')}.download_ready"
    if not model_cache.exists() or not model_marker.exists():
        from huggingface_hub import snapshot_download

        LOGGER.info("Neodragon weights missing; downloading %s into %s", model_id, cache)
        _with_file_lock(
            model_marker,
            lambda: (snapshot_download(model_id, cache_dir=str(cache)), model_marker.touch()),
        )

    from huggingface_hub import snapshot_download

    local_model_path = snapshot_download(model_id, cache_dir=str(cache), local_files_only=model_cache.exists())
    return str(repo), str(cache), str(local_model_path)
