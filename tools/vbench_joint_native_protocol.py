"""CPU helpers for the paired Joint V2 / released-native benchmark."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


FORMAT = "mobileov_joint_v2_native_quicksr_v1"
HF_REPO = "Amshaker/Mobile-OV"
HF_REVISION = "fddde4eff65780dc106987473397b101e4fa15a8"
JOINT_FILE = "mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt"
JOINT_SHA = "f70909cccdcfc6f40beb5fafb11acdadcdd7b546e6fff2347182ec0a2ebaa14f"
IMAGE_FILE = "dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt"
IMAGE_SHA = "371a393cfee257436124ea3c1cc0919f2ade28f7928608a1afafd74fe5052126"
NATIVE_REVISION = "5f296764e442d7261c53fcf9367a6b0f01e22f7f"
SYSTEMS = ("joint_v2", "native_hybrid")
SEED_MAPPINGS = ("legacy", "extendable")
SEED_STRIDE = 1_000_003
DIMENSIONS = (
    "subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree",
    "aesthetic_quality", "imaging_quality", "temporal_flickering", "object_class",
    "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style",
    "overall_consistency", "human_action", "appearance_style",
)
# Relative costs from the saved A100 4-seed + QuickSR run; used only for scheduling.
SCORE_COSTS = (3.3, 1.4, 5.4, 9.6, 1.8, 5.6, .8, 16.5, 18.9, 22, 18, 5.2, 1.6, 1.5, 2.4, 7.2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".benchmark.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another benchmark owns {output}") from error
        yield


def load_prompt_rows(path, max_prompts=0):
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("VBench full-info must be a list")
    prompts, seen = [], set()
    for row in rows:
        value = row.get("prompt_en")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing prompt_en")
        prompt = " ".join(value.split())
        if any(char in prompt for char in ("/", "\\", "\0")):
            raise ValueError(f"Unsafe VBench filename: {prompt!r}")
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    if not prompts:
        raise ValueError("Empty prompt list")
    if max_prompts:
        prompts = prompts[:max_prompts]
    selected = set(prompts)
    # Keep ALL dimension memberships, including duplicate prompt rows.
    filtered = [{**row, "prompt_en": " ".join(row["prompt_en"].split())}
                for row in rows if " ".join(row["prompt_en"].split()) in selected]
    return prompts, filtered


def sample_items(prompts, samples, seed, seed_mapping="legacy"):
    if seed_mapping not in SEED_MAPPINGS:
        raise ValueError(f"Unknown seed mapping: {seed_mapping}")
    if seed_mapping == "extendable" and len(prompts) >= SEED_STRIDE:
        raise ValueError("Too many prompts for the fixed seed stride")
    for index, prompt in enumerate(prompts):
        for sample in range(samples):
            filename = f"{prompt}-{sample}.mp4"
            if len(filename.encode("utf-8")) > 255:
                raise ValueError(f"VBench filename exceeds filesystem limit: {filename}")
            sample_seed = (seed + index + sample * SEED_STRIDE if seed_mapping == "extendable"
                           else seed + index * samples + sample)
            yield dict(prompt=prompt, prompt_index=index, sample_index=sample,
                       anchor_seed=sample_seed, video_seed=sample_seed,
                       filename=filename, key=f"{index:04d}_{sample}")


def shard_items(items, rank, workers):
    if workers < 1 or not 0 <= rank < workers:
        raise ValueError("Invalid worker rank/count")
    return [item for index, item in enumerate(items) if index % workers == rank]


def score_assignments(workers, systems=SYSTEMS):
    if workers < 1:
        raise ValueError("workers must be positive")
    if not systems or len(set(systems)) != len(systems) or any(system not in SYSTEMS for system in systems):
        raise ValueError("Invalid selected systems")
    groups, costs = [[] for _ in range(workers)], [0.] * workers
    tasks = [(system, index) for system in systems for index in range(len(DIMENSIONS))]
    for system, index in sorted(tasks, key=lambda item: -SCORE_COSTS[item[1]]):
        rank = min(range(workers), key=lambda candidate: (costs[candidate], candidate))
        groups[rank].append((system, index))
        costs[rank] += SCORE_COSTS[index]
    return groups


def bind_contract(output, contract):
    target = output / "run_contract.json"
    if target.exists():
        previous = read_json(target)
        if previous != contract:
            changed = sorted(key for key in set(previous) | set(contract)
                             if previous.get(key) != contract.get(key))
            raise ValueError(f"Resume contract differs: {changed}. Use a new RUN_ROOT.")
    else:
        if any(output.glob("*/videos/*.mp4")) or any(output.glob("*/scores/**/*.json")):
            raise ValueError("Cannot reuse results without their run contract")
        atomic_json(target, contract)
    return fingerprint(contract)


def validate_payload(payload, kind):
    import torch

    if kind == "joint":
        if payload.get("step") != 10000 or payload.get("teacher_stack", {}).get("name") != "multistep":
            raise ValueError("Expected Joint V2 step 10000, multistep monolithic")
        components = ("bridge", "dit")
    else:
        config = payload.get("config", {})
        if payload.get("step") != 160000 or not isinstance(config, dict) or config.get("training_version") != "v11_balanced":
            raise ValueError("Expected V11-balanced image bridge step 160000")
        components = ("bridge",)
    for component in components:
        state = payload.get(component)
        if not isinstance(state, dict) or not state or any(not isinstance(t, torch.Tensor) for t in state.values()):
            raise ValueError(f"Missing/invalid trained {component} weights")


def ensure_checkpoint(path, filename, expected_sha, kind):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            from huggingface_hub import hf_hub_download

            source = hf_hub_download(HF_REPO, filename, revision=HF_REVISION, force_download=True)
            temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
            try:
                shutil.copyfile(source, temporary)
                if sha256(temporary) != expected_sha:
                    raise ValueError(f"Downloaded checkpoint SHA mismatch: {filename}")
                candidate = torch.load(temporary, map_location="cpu", mmap=True, weights_only=False)
                validate_payload(candidate, kind)
                del candidate
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        actual = sha256(path)
        if actual != expected_sha:
            raise ValueError(f"Checkpoint SHA mismatch: {path}. Existing file is NOT overwritten.")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        validate_payload(payload, kind)
        del payload
    return {"path": str(path.resolve()), "sha256": actual}


def decode_video(path, shape):
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or abs(fps - shape["fps"]) > .05:
            return False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (shape["height"], shape["width"]):
                return False
            count += 1
    finally:
        capture.release()
    return count == shape["frames"]


def cached_video(video, record_path, item, system, contract_sha, shape):
    try:
        record = read_json(record_path)
        return (record.get("sample") == item and record.get("system") == system
                and record.get("contract_sha256") == contract_sha
                and record.get("shape") == shape and video.is_file()
                and sha256(video) == record.get("video_sha256")
                and decode_video(video, shape))
    except (OSError, ValueError, KeyError):
        return False


def aggregate_scores(summaries, contract):
    if set(summaries) != set(SYSTEMS):
        raise ValueError("Both systems must have complete scores")
    for name, row in summaries.items():
        if set(row.get("raw", {})) != set(DIMENSIONS):
            raise ValueError(f"Incomplete dimensions for {name}")
        numbers = [row["quality_score"], row["semantic_score"], row["total_score"], *row["raw"].values()]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"Non-finite scores for {name}")
        if not math.isclose(row["total_score"], .8 * row["quality_score"] + .2 * row["semantic_score"], abs_tol=1e-9):
            raise ValueError(f"Incorrect aggregate for {name}")
    joint, native = summaries["joint_v2"], summaries["native_hybrid"]
    return {
        "status": "complete", "protocol": contract,
        "scores": summaries,
        "joint_minus_native_percentage_points": {
            **{key: 100 * (joint[key] - native[key]) for key in ("quality_score", "semantic_score", "total_score")},
            "raw_dimensions": {key: 100 * (joint["raw"][key] - native["raw"][key]) for key in DIMENSIONS},
        },
        "scope": "Matched end-to-end protocol and QuickSR. First-frame generators and denoising schedules differ by design. Not a conditioner-only ablation, not a paper-score reproduction, and not a mobile-latency comparison.",
    }
