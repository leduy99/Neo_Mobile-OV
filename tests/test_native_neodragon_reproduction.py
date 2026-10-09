from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option
from tools import reproduce_native_neodragon_vbench as runner
from tools import generate_native_neodragon_release as worker
from tools import vbench_joint_native_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_native_neodragon_vbench_1node8gpu.sbatch"


def inference_script(tmp_path):
    path = tmp_path / "inference.py"
    path.write_text('prompt_modifier = ", cinematic" if args.use_cinematic_prompt_modifier else ", hyper quality, Ultra HD, 8K"\n')
    return path


def template():
    return dict(generation_code_sha256="original", selected_systems=["native_hybrid"],
                prompts=["panda"], samples_per_prompt=5, seed=2026, seed_mapping="extendable",
                anchor=dict(width=1024, height=640), base_video=dict(width=512, height=320, frames=49, fps=24),
                output=dict(width=1024, height=640, frames=49, fps=24),
                quicksr=dict(variant="medium", scale=2, dtype="fp16", batch_frames=4),
                weights=dict(joint=None, image=None, smolvlm2=None, quicksr=dict(path="qsr")))


def test_extracts_original_cli_suffix_and_not_optional_cinematic_suffix(tmp_path):
    assert runner.released_prompt_modifier(inference_script(tmp_path)) == ", hyper quality, Ultra HD, 8K"
    source = ROOT / "checkpoints/neodragon_repo/scripts/inference_neodragon.py"
    if source.exists():
        assert runner.released_prompt_modifier(source) == ", hyper quality, Ultra HD, 8K"


def test_changed_upstream_cli_layout_fails_instead_of_guessing_suffix(tmp_path):
    path = tmp_path / "unknown.py"
    path.write_text('modifier = "changed"\n')
    with pytest.raises(ValueError, match="Cannot identify"):
        runner.released_prompt_modifier(path)


def test_default_is_native_only_and_never_reuses_matched_cinematic_results():
    args = runner.parse_args([])
    assert args.systems == ["native_hybrid"] and args.workers == 8 and args.samples_per_prompt == 5
    assert args.output_dir.name == "vbench_native_neodragon_release_5x"
    with pytest.raises(ValueError, match="only runs native_hybrid"):
        runner.parse_args(["--systems", "joint_v2"])
    with pytest.raises(ValueError, match="Do not reuse"):
        runner.parse_args(["--seed-mapping", "extendable", "--reuse-1x-run", "native_hybrid=old"])


def test_contract_records_seed_difference_and_keeps_two_paired_variants(tmp_path):
    base = template()
    original = deepcopy(base)
    path = inference_script(tmp_path)
    contracts = runner.native_contracts(base, runner.released_prompt_modifier(path), path)
    assert base == original
    sr, raw = contracts["quicksr"], contracts["no_quicksr"]
    assert sr["generation_pair_sha256"] == raw["generation_pair_sha256"]
    assert sr["output"]["width"] == 1024 and raw["output"]["width"] == 512
    policy = sr["release_policy"]
    assert policy["image_bridge"] is False and policy["video_bridge"] is False
    assert policy["native_text_conditioner"] is True
    assert "not between image and video" in policy["seed_reset"]
    assert policy["public_quicksr_variant_matches_paper_verified"] is False
    assert sr["upstream_inference_script"]["sha256"] == protocol.sha256(path)
    assert sr["generation_code_sha256"] == protocol.sha256(runner.WORKER)
    assert sr["dual_export_helper_sha256"] == protocol.sha256(ROOT / "tools/generate_vbench_joint_v2_dual_export.py")
    assert sr["weights"]["joint"] is None and sr["weights"]["image"] is None


def test_capture_is_transparent_and_passes_kwargs_without_rng_side_effects():
    image = Image.new("RGB", (2, 2), "red")
    calls = []

    class ImagePipeline:
        name = "native"

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(images=[image])

    capture = worker.CaptureFirstFrame(ImagePipeline())
    torch.manual_seed(2026)
    before = torch.get_rng_state().clone()
    output = capture(prompt="panda", num_images_per_prompt=1)
    assert output.images[0] is image and capture.image is image and capture.calls == 1
    assert calls == [((), dict(prompt="panda", num_images_per_prompt=1))]
    assert torch.equal(before, torch.get_rng_state()) and capture.name == "native"


def test_full_native_call_preserves_image_to_video_rng_consumption(monkeypatch):
    seeds = []

    def seed_once(seed):
        seeds.append(seed)
        torch.manual_seed(seed)

    monkeypatch.setattr(worker.generation, "seed_all", seed_once)

    class ImagePipeline:
        def __call__(self, **kwargs):
            assert "generator" not in kwargs
            self.draw = torch.rand(()).item()
            return SimpleNamespace(images=[Image.new("RGB", (2, 2), "red")])

    image = ImagePipeline()
    capture = worker.CaptureFirstFrame(image)

    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            capture(prompt=kwargs["prompt"] + kwargs["prompt_modifier"], num_images_per_prompt=1)
            self.draw = torch.rand(()).item()
            return ["video"]

    pipeline = Pipeline()
    item = {"prompt": "panda", "video_seed": 2026}
    frames, anchor = worker.generate_clip(pipeline, capture, item, ", hyper quality, Ultra HD, 8K")
    torch.manual_seed(2026)
    expected = torch.rand(2)
    assert seeds == [2026] and capture.calls == 1
    assert image.draw == expected[0].item() and pipeline.draw == expected[1].item()
    assert pipeline.kwargs == dict(prompt="panda", height=320, width=512, num_frames=49,
                                   prompt_modifier=", hyper quality, Ultra HD, 8K", profile=False)
    assert frames == ["video"] and anchor is capture.image


def test_native_pipeline_must_generate_its_own_first_image(monkeypatch):
    monkeypatch.setattr(worker.generation, "seed_all", lambda _: None)
    capture = SimpleNamespace(image=None, calls=0)
    with pytest.raises(ValueError, match="exactly one SSD1B"):
        worker.generate_clip(lambda **kw: ["video"], capture, dict(prompt="panda", video_seed=2026), "suffix")


def test_prepare_certifies_both_contracts_and_unchanged_resume(tmp_path, monkeypatch):
    source = tmp_path / "native/scripts"
    source.mkdir(parents=True)
    inference_script(tmp_path).rename(source / "inference_neodragon.py")

    def prepare(args):
        assert args.systems == ["native_hybrid"]
        protocol.atomic_json(args.output_dir / "vbench_info.json", [{"prompt_en": "panda", "dimension": ["scene"]}])
        protocol.atomic_json(args.output_dir / "sample_plan.json", {"samples": []})
        return template()

    monkeypatch.setattr(runner.core, "prepare", prepare)
    args = SimpleNamespace(output_dir=tmp_path / "results", native_repo=source.parent, systems=["native_hybrid"])
    contract = runner.prepare(args)
    assert contract["format"] == "native_neodragon_release_reproduction_v1"
    assert runner.prepare(args) == contract
    for variant in runner.dual.VARIANTS:
        folder = args.output_dir / variant
        assert protocol.fingerprint(protocol.read_json(folder / "run_contract.json")) == contract["variants"][variant]


def test_paper_reference_report_uses_only_published_dimensions_and_preserves_scores(tmp_path):
    scores = dict(total_score=.8161, quality_score=.8368, semantic_score=.7336,
                  raw={name: value / 100 for name, value in runner.PAPER_SCORES.items() if not name.endswith("_score")})
    path = tmp_path / "quicksr_ablation.json"
    protocol.atomic_json(path, {"scores": {"quicksr": {"native_hybrid": scores}}})
    before = protocol.sha256(path)
    runner.paper_comparison(tmp_path)
    report = protocol.read_json(tmp_path / "paper_reference_comparison.json")
    assert set(report["rows"]) == set(runner.PAPER_SCORES)
    assert report["rows"]["total_score"]["difference_pp"] == pytest.approx(0)
    assert len(report["unavailable_paper_dimensions"]) == 10
    assert protocol.sha256(path) == before
    assert "not a paper-exact" in (tmp_path / "paper_reference_comparison.md").read_text()


def test_sbatch_requests_eight_gpus_and_native_only_defaults(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    root, _, run = launch
    result, argv = run()
    assert result.returncode == 0, result.stderr
    assert "tools/reproduce_native_neodragon_vbench.py" in argv
    assert option(argv, "--systems") == "native_hybrid" and "joint_v2" not in argv
    assert "--joint-checkpoint" not in argv and "--image-checkpoint" not in argv
    assert option(argv, "--samples-per-prompt") == "5" and option(argv, "--workers") == "8"
    assert option(argv, "--seed-mapping") == "extendable"
    assert option(argv, "--output-dir") == "output/vbench_native_neodragon_release_5x"
    assert "mock report" in result.stdout
    text = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:8" in text and "#SBATCH --partition=berzelius" in text


def test_native_sbatch_smoke_can_generate_one_sample_without_scoring(launch, monkeypatch):
    monkeypatch.setattr(launch_helpers, "SCRIPT", SCRIPT)
    _, _, run = launch
    result, argv = run(MAX_PROMPTS="2", SAMPLES_PER_PROMPT="1", NATIVE_MODEL_DIR="/weights/native snapshot")
    assert result.returncode == 0, result.stderr
    assert option(argv, "--max-prompts") == "2" and option(argv, "--native-model-dir") == "/weights/native snapshot"
    assert option(argv, "--output-dir").endswith("_1x_smoke2")
