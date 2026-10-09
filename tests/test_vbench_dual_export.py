from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tools import benchmark_joint_v2_quicksr_ablation as runner
from tools import generate_vbench_joint_v2_dual_export as worker
from tools import vbench_joint_native_protocol as protocol


def template():
    return dict(prompts=["panda"], selected_systems=["joint_v2"], samples_per_prompt=1,
                seed=2026, seed_mapping="extendable", protocol_name="paired1x",
                generation_code_sha256="generation helper",
                anchor=dict(width=32, height=16), base_video=dict(width=32, height=16, frames=3, fps=24),
                output=dict(width=64, height=32, frames=3, fps=24),
                quicksr=dict(variant="medium", scale=2, dtype="fp16", batch_frames=4))


def fixture_pair(tmp_path):
    contracts = runner.variant_contracts(template())
    parent = dict(selected_systems=["joint_v2"], variants={name: protocol.fingerprint(value)
                 for name, value in contracts.items()})
    protocol.atomic_json(tmp_path / "run_contract.json", parent)
    for name, contract in contracts.items():
        protocol.atomic_json(tmp_path / name / "run_contract.json", contract)
    item = list(protocol.sample_items(["panda"], 1, 2026, "extendable"))[0]
    folder = tmp_path / "quicksr/joint_v2"
    anchor = folder / "anchors" / f"{item['key']}.png"
    worker.generation.save_anchor(Image.new("RGB", (32, 16), "red"), anchor,
                                 folder / "anchor_records" / f"{item['key']}.json",
                                 item, protocol.fingerprint(contracts["quicksr"]))
    array = np.random.default_rng(2026).integers(0, 256, (3, 16, 32, 3), dtype=np.uint8)
    kwargs = dict(output=tmp_path, contracts=contracts, system="joint_v2", item=item,
                  anchor_path=anchor, model=torch.nn.Upsample(scale_factor=2, mode="nearest"), device=torch.device("cpu"))
    return array, kwargs


def test_variant_contracts_keep_generation_identical_and_state_resolution_difference():
    base = template()
    original = deepcopy(base)
    variants = runner.variant_contracts(base)
    assert base == original
    sr, raw = variants["quicksr"], variants["no_quicksr"]
    assert sr["output"] == base["output"] and raw["output"] == base["base_video"]
    assert raw["quicksr"] == {"enabled": False, "resize": False}
    assert sr["generation_pair_sha256"] == raw["generation_pair_sha256"]
    differing = {key for key in sr if sr[key] != raw[key]}
    assert differing == {"postprocessing_variant", "quicksr", "output", "scope"}


def test_actual_mp4_exports_keep_same_raw_frames_and_both_decode(tmp_path):
    array, kwargs = fixture_pair(tmp_path)
    before = array.copy()
    worker.export_pair(array, **kwargs)
    assert np.array_equal(array, before)
    assert worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])
    for variant, contract in kwargs["contracts"].items():
        record = protocol.read_json(tmp_path / variant / "joint_v2/records/0000_0.json")
        assert record["base_frames_sha256"] == worker.frames_digest(before)
        assert record["shape"] == contract["output"]
        assert protocol.decode_video(tmp_path / variant / "joint_v2/videos/panda-0.mp4", contract["output"])
    runner.verify_generation(tmp_path)
    assert protocol.read_json(tmp_path / "generation_summary.json")["same_generated_frames"] is True


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_a_missing_or_corrupted_variant_invalidates_the_entire_pair(tmp_path, variant):
    array, kwargs = fixture_pair(tmp_path)
    worker.export_pair(array, **kwargs)
    path = tmp_path / variant / "joint_v2/videos/panda-0.mp4"
    path.write_bytes(b"corrupted")
    assert not worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])


def test_different_base_frame_provenance_is_rejected(tmp_path):
    array, kwargs = fixture_pair(tmp_path)
    worker.export_pair(array, **kwargs)
    path = tmp_path / "no_quicksr/joint_v2/records/0000_0.json"
    record = protocol.read_json(path)
    record["base_frames_sha256"] = "0" * 64
    protocol.atomic_json(path, record)
    assert not worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])
    with pytest.raises(ValueError, match="share generated frames"):
        runner.verify_generation(tmp_path)


def test_second_export_failure_never_certifies_a_partial_pair(tmp_path, monkeypatch):
    from diffusers import utils

    array, kwargs = fixture_pair(tmp_path)
    original = utils.export_to_video
    calls = []

    def fail_second(*args, **kw):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("encoder interrupted")
        return original(*args, **kw)

    monkeypatch.setattr(utils, "export_to_video", fail_second)
    with pytest.raises(RuntimeError, match="encoder interrupted"):
        worker.export_pair(array, **kwargs)
    assert not list(tmp_path.glob("*/joint_v2/records/*.json"))
    assert not list(tmp_path.glob("*/joint_v2/videos/.*.tmp.*.mp4"))
    assert not worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])
    monkeypatch.setattr(utils, "export_to_video", original)
    worker.export_pair(array, **kwargs)
    assert worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])


def test_same_anchor_is_hardlinked_without_modifying_original(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"anchor")
    target = tmp_path / "other/anchor.png"
    worker.copy_anchor(source, target)
    assert target.read_bytes() == source.read_bytes() == b"anchor"
    assert target.stat().st_ino == source.stat().st_ino


def test_two_variants_and_two_models_cover_64_disjoint_scoring_tasks():
    groups = runner.score_assignments(8, protocol.SYSTEMS)
    tasks = [task for group in groups for task in group]
    assert all(groups) and len(tasks) == len(set(tasks)) == 64
    assert set(tasks) == {(variant, system, i) for variant in runner.VARIANTS
                          for system in protocol.SYSTEMS for i in range(16)}
    assert len([task for group in runner.score_assignments(8, ["joint_v2"]) for task in group]) == 32


def test_score_workers_never_mix_variant_input_and_output_paths(tmp_path, monkeypatch):
    protocol.atomic_json(tmp_path / "run_contract.json", {"selected_systems": list(protocol.SYSTEMS)})
    protocol.atomic_json(tmp_path / "quicksr/run_contract.json", {"samples_per_prompt": 5})
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kw: calls.append(argv))
    runner.score_worker(SimpleNamespace(output_dir=tmp_path, workers=8, rank=0, vbench_repo=tmp_path))
    assert calls
    for argv in calls:
        videos = Path(argv[argv.index("--videos") + 1])
        output = Path(argv[argv.index("--output") + 1])
        info = Path(argv[argv.index("--full-info") + 1])
        assert videos.parent == output.parents[1]
        assert info.parent == videos.parents[1]
        assert argv[argv.index("--expected-samples-per-prompt") + 1] == "5"


def test_summary_reports_quicksr_gain_without_inventing_native_results():
    score = {key: .8 for key in ("quality_score", "semantic_score", "total_score")}
    score["raw"] = {dimension: .5 for dimension in protocol.DIMENSIONS}
    raw = deepcopy(score)
    raw["total_score"] = .75
    report, markdown = runner.ablation_report({"quicksr": {"joint_v2": score}, "no_quicksr": {"joint_v2": raw}},
                                             {"selected_systems": ["joint_v2"]})
    assert report["quicksr_minus_raw_percentage_points"]["joint_v2"]["total_score"] == pytest.approx(5)
    assert "native_hybrid" not in markdown and "512x320" in markdown and "1024x640" in markdown


def test_old_upscaled_only_runs_cannot_be_reused_as_original_rgb_clips():
    with pytest.raises(ValueError, match="must generate both anew"):
        runner.parse_args(["--seed-mapping", "extendable", "--reuse-1x-run", "joint_v2=old"])
    args = runner.parse_args([])
    assert args.output_dir.name == "vbench_joint_v2_vs_native_5x_dual"
    assert args.samples_per_prompt == 5 and args.workers == 8


def test_prepare_binds_both_variants_and_supports_unchanged_resume(tmp_path, monkeypatch):
    def asset_prepare(args):
        protocol.atomic_json(args.output_dir / "vbench_info.json", [{"prompt_en": "panda", "dimension": ["scene"]}])
        protocol.atomic_json(args.output_dir / "sample_plan.json", {"contract_sha256": "original", "samples": []})
        return template()

    monkeypatch.setattr(runner.core, "prepare", asset_prepare)
    args = SimpleNamespace(output_dir=tmp_path)
    contract = runner.prepare(args)
    assert runner.prepare(args) == contract
    for variant in runner.VARIANTS:
        folder = tmp_path / variant
        assert protocol.fingerprint(protocol.read_json(folder / "run_contract.json")) == contract["variants"][variant]
        assert protocol.read_json(folder / "sample_plan.json")["contract_sha256"] == contract["variants"][variant]


def test_real_quicksr_weights_work_with_cpu_export_path(tmp_path):
    from new_mobile_ov.generation.quicksrnet import load_quicksrnet

    checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/quicksrnet/quicksrnet_medium_2x_float32.pth.tar"
    if not checkpoint.is_file():
        pytest.skip("Optional local QuickSR weights are not available")
    array, kwargs = fixture_pair(tmp_path)
    model, _ = load_quicksrnet(variant="medium", scale=2, checkpoint=str(checkpoint),
                              device=torch.device("cpu"), dtype=torch.float16)
    kwargs["model"] = model
    worker.export_pair(array, **kwargs)
    assert worker.paired_cached(tmp_path, kwargs["contracts"], "joint_v2", kwargs["item"])


def test_full_summary_merges_all_four_results_and_writes_comparisons(tmp_path, monkeypatch):
    scores = dict(quality_score=.8, semantic_score=.6, total_score=.76,
                  raw={dimension: .5 for dimension in protocol.DIMENSIONS})
    module = ModuleType("evaluate_vbench_resume")
    module.tabulate = lambda _: deepcopy(scores)
    monkeypatch.setitem(sys.modules, "evaluate_vbench_resume", module)
    monkeypatch.setattr(runner, "verify_generation", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kw: None)
    protocol.atomic_json(tmp_path / "run_contract.json", {"selected_systems": list(protocol.SYSTEMS)})
    protocol.atomic_json(tmp_path / "score_assets.json", [])
    for variant in runner.VARIANTS:
        protocol.atomic_json(tmp_path / variant / "run_contract.json", {"scope": f"{variant} actual resolution"})
        for system in protocol.SYSTEMS:
            for dimension in protocol.DIMENSIONS:
                protocol.atomic_json(tmp_path / variant / system / "scores" / dimension / "vbench" /
                                     f"{dimension}_eval_results.json", {dimension: [.5, []]})
    runner.summarize(SimpleNamespace(output_dir=tmp_path))
    for variant in runner.VARIANTS:
        report = protocol.read_json(tmp_path / variant / "comparison.json")
        assert report["scope"] == f"{variant} actual resolution"
        assert set(report["scores"]) == set(protocol.SYSTEMS)
    report = protocol.read_json(tmp_path / "quicksr_ablation.json")
    assert report["quicksr_minus_raw_percentage_points"]["joint_v2"]["total_score"] == 0
    assert set(report["scores"]) == set(runner.VARIANTS)
    markdown = (tmp_path / "quicksr_ablation.md").read_text()
    assert "native_hybrid: No QuickSR (512x320)" in markdown
    assert "joint_v2: QuickSR 2x (1024x640)" in markdown
