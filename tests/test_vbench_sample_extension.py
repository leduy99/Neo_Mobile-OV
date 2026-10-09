from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from tools import benchmark_joint_v2_native as runner
from tools import vbench_joint_native_protocol as protocol
from tools import reuse_vbench_1x_samples as reuse


def test_extendable_mapping_keeps_every_existing_sample_zero():
    prompts = [f"prompt {i}" for i in range(944)]
    old = list(protocol.sample_items(prompts, 1, 2026))
    five = list(protocol.sample_items(prompts, 5, 2026, "extendable"))
    assert [item for item in five if item["sample_index"] == 0] == old
    assert len({item["video_seed"] for item in five}) == 4720
    assert five[1]["video_seed"] == 2026 + protocol.SEED_STRIDE
    for workers in (2, 8):
        groups = [protocol.shard_items(five, rank, workers) for rank in range(workers)]
        assert len({item["key"] for group in groups for item in group}) == 4720


def test_reuse_flags_require_nonchanging_seed_mapping():
    with pytest.raises(SystemExit):
        runner.parse_args(["--reuse-1x-run", "joint_v2=old"])
    args = runner.parse_args(["--seed-mapping", "extendable", "--reuse-1x-run", "joint_v2=old"])
    assert args.samples_per_prompt == 5


def fixture_runs(tmp_path):
    source, output = tmp_path / "one", tmp_path / "five"
    source.mkdir()
    output.mkdir()
    shared = {name: "same" for name in reuse.COMMON_FIELDS}
    shared.update(prompts=["panda", "car"], seed=2026, output=dict(width=1024, height=640, frames=49, fps=24),
                  selected_systems=["joint_v2"], samples_per_prompt=1,
                  generation_code_sha256=reuse.REUSABLE_1X_GENERATOR,
                  systems=dict(joint_v2="same generator"),
                  weights={name: dict(sha256="same") for name in ("joint", "image", "smolvlm2", "quicksr")},
                  assets=dict(native_source_sha256="code", native_source_commit="commit", native_model_dir="model",
                              native_files=[dict(file="vae/weight", size=42, sha256="asset")], dreamlite_files=[]))
    protocol.atomic_json(source / "run_contract.json", shared)
    protocol.atomic_json(source / "generation_summary.json", dict(status="complete", contract_sha256=protocol.fingerprint(shared)))
    for item in protocol.sample_items(shared["prompts"], 1, shared["seed"]):
        folder = source / "joint_v2"
        video, anchor = folder / "videos" / item["filename"], folder / "anchors" / f"{item['key']}.png"
        video.parent.mkdir(parents=True, exist_ok=True)
        anchor.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(item["prompt"].encode())
        anchor.write_bytes(b"anchor")
        protocol.atomic_json(folder / "records" / f"{item['key']}.json", dict(sample=item, system="joint_v2",
            contract_sha256=protocol.fingerprint(shared), shape=shared["output"],
            video_sha256=protocol.sha256(video), anchor_sha256=protocol.sha256(anchor)))
        protocol.atomic_json(folder / "anchor_records" / f"{item['key']}.json", dict(sample=item,
            contract_sha256=protocol.fingerprint(shared), sha256=protocol.sha256(anchor)))
    target = deepcopy(shared)
    target.update(samples_per_prompt=5, seed_mapping="extendable", generation_code_sha256="successor")
    return source, output, shared, target


def test_existing_media_is_byte_identical_and_new_metadata_has_provenance(tmp_path):
    source, output, old, target = fixture_runs(tmp_path)
    report = reuse.reuse_samples(output, target, [f"joint_v2={source}"])
    assert report["reused_videos"] == 2 and report["old_scores_reused"] is False
    for item in protocol.sample_items(target["prompts"], 1, target["seed"]):
        copied = output / "joint_v2/videos" / item["filename"]
        assert copied.read_bytes() == (source / "joint_v2/videos" / item["filename"]).read_bytes()
        record = protocol.read_json(output / "joint_v2/records" / f"{item['key']}.json")
        assert record["sample"] == item and record["contract_sha256"] == protocol.fingerprint(target)
        assert record["reused_from_1x"]["source_contract_sha256"] == protocol.fingerprint(old)
    assert not (output / "joint_v2/scores").exists()
    assert reuse.reuse_samples(output, target, [f"joint_v2={source}"])["reused_videos"] == 2


@pytest.mark.parametrize("change", ["seed", "prompts", "quicksr", "output", "weights", "generator"])
def test_changed_generation_settings_are_rejected(tmp_path, change):
    source, output, _, target = fixture_runs(tmp_path)
    if change == "seed":
        target["seed"] += 1
    elif change == "prompts":
        target["prompts"].reverse()
    elif change == "weights":
        target["weights"]["joint"]["sha256"] = "different"
    elif change == "generator":
        old = protocol.read_json(source / "run_contract.json")
        old["generation_code_sha256"] = "unknown"
        protocol.atomic_json(source / "run_contract.json", old)
        protocol.atomic_json(source / "generation_summary.json", dict(status="complete", contract_sha256=protocol.fingerprint(old)))
    else:
        target[change] = "different"
    with pytest.raises(ValueError):
        reuse.reuse_samples(output, target, [f"joint_v2={source}"])
    assert not (output / "joint_v2/videos").exists()


def test_source_corruption_is_caught_before_any_import(tmp_path):
    source, output, _, target = fixture_runs(tmp_path)
    (source / "joint_v2/videos/car-0.mp4").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Unverified source sample"):
        reuse.reuse_samples(output, target, [f"joint_v2={source}"])
    assert not (output / "joint_v2/videos").exists()


def test_conflicting_target_is_never_overwritten(tmp_path):
    source, output, _, target = fixture_runs(tmp_path)
    path = output / "joint_v2/videos/panda-0.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"different existing video")
    with pytest.raises(ValueError, match="conflicting sample"):
        reuse.reuse_samples(output, target, [f"joint_v2={source}"])
    assert path.read_bytes() == b"different existing video"


def test_local_expansion_job_respects_current_quota_and_preserves_both_sources():
    script = Path(__file__).resolve().parents[1] / "scripts/vbench_joint_v2_native_extend5x_local.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "#SBATCH --gres=gpu:2" in text and "#SBATCH --time=16:00:00" in text
    assert "--seed-mapping extendable" in text and "--samples-per-prompt 5" in text
    assert text.count("--reuse-1x-run") == 2
