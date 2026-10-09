from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from tools import benchmark_first_frame_swap as runner
from tools import vbench_joint_native_protocol as protocol
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option

ROOT = Path(__file__).resolve().parents[1]


def fixture_baseline(tmp_path):
    template = dict(prompts=["panda"], samples_per_prompt=1, seed=2026, seed_mapping="extendable",
                    anchor=dict(width=32,height=16), base_video=dict(width=32,height=16,frames=3,fps=24),
                    output=dict(width=64,height=32,frames=3,fps=24), full_info_sha256="info",
                    compat_code_sha256="compat",score_code_sha256="scores",vbench_source_commit="vbench",
                    systems={"joint_v2":"monolithic 20/10 CFG7/5", "native_hybrid":"hybrid 1-1-1 noCFG"},
                    quicksr=dict(variant="medium",scale=2,dtype="fp16",batch_frames=4),
                    image_config=dict(sha256="image"),video_config=dict(sha256="video"),
                    weights={name:dict(sha256=name) for name in ("joint","image","smolvlm2","quicksr")},
                    assets=dict(native_source_commit="native",native_source_sha256="code",native_files=[],dreamlite_files=[]),
                    generation_code_sha256="helper")
    baseline = deepcopy(template)
    baseline["quicksr"]["enabled"] = True
    baseline["generation_helper_code_sha256"] = template["generation_code_sha256"]
    parent = dict(format="mobileov_joint_native_dual_export_v1",selected_systems=list(protocol.SYSTEMS),
                  variants=dict(quicksr=protocol.fingerprint(baseline)))
    root = tmp_path / "baseline"
    protocol.atomic_json(root / "run_contract.json", parent)
    protocol.atomic_json(root / "quicksr/run_contract.json", baseline)
    protocol.atomic_json(root / "generation_summary.json", dict(status="complete",contract_sha256=protocol.fingerprint(parent)))
    item = next(protocol.sample_items(["panda"],1,2026,"extendable"))
    for system,color in (("joint_v2","red"),("native_hybrid","blue")):
        folder = root / "quicksr" / system
        runner.generation.save_anchor(Image.new("RGB",(32,16),color),folder/"anchors/0000_0.png",
                                      folder/"anchor_records/0000_0.json",item,protocol.fingerprint(baseline))
    return root,template,baseline,parent,item


@pytest.mark.parametrize("case,source", [("joint_v2_ssd1b","native_hybrid"),("native_hybrid_dreamlite_v11","joint_v2")])
def test_correct_opposite_image_is_reused_without_changing_original(tmp_path,case,source):
    root,template,baseline,parent,item = fixture_baseline(tmp_path)
    _,records = runner.source_anchors(template,root,case)
    assert Path(records[0]["path"]) == root/"quicksr"/source/"anchors/0000_0.png"
    contract = dict(anchor_swap=dict(source_anchors=records,video_system=runner.SWAPS[case]["video"]))
    target = runner.supplied_anchor(tmp_path/"new",contract,item)
    assert protocol.sha256(target) == protocol.sha256(Path(records[0]["path"])) == records[0]["sha256"]
    assert protocol.read_json(target.parent.parent/"anchor_records/0000_0.json")["contract_sha256"] == protocol.fingerprint(contract)
    assert Path(records[0]["path"]).exists()


@pytest.mark.parametrize("field",["seed","seed_mapping","samples_per_prompt","anchor","systems","weights","assets"])
def test_changed_seed_schedule_weights_or_assets_are_rejected(tmp_path,field):
    root,template,baseline,parent,_ = fixture_baseline(tmp_path)
    if field == "weights": template["weights"]["joint"]["sha256"] = "changed"
    elif field == "assets": template["assets"]["native_source_commit"] = "changed"
    else: template[field] = "changed"
    with pytest.raises(ValueError): runner.source_anchors(template,root,"joint_v2_ssd1b")


def test_native_release_results_cannot_replace_matched_baseline(tmp_path):
    root,template,baseline,parent,_ = fixture_baseline(tmp_path)
    parent["format"] = "native_neodragon_release_reproduction_v1"
    protocol.atomic_json(root/"run_contract.json",parent)
    protocol.atomic_json(root/"generation_summary.json",dict(status="complete",contract_sha256=protocol.fingerprint(parent)))
    with pytest.raises(ValueError,match="NOT native-release"):
        runner.source_anchors(template,root,"joint_v2_ssd1b")


def test_missing_or_corrupted_original_png_is_not_regenerated(tmp_path):
    root,template,baseline,parent,_ = fixture_baseline(tmp_path)
    image = root/"quicksr/native_hybrid/anchors/0000_0.png"
    image.write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="Unverified"):
        runner.source_anchors(template,root,"joint_v2_ssd1b")
    image.unlink()
    with pytest.raises(FileNotFoundError,match="do not regenerate"):
        runner.source_anchors(template,root,"joint_v2_ssd1b")


def test_changed_anchor_after_preflight_is_rejected(tmp_path):
    root,template,baseline,parent,item = fixture_baseline(tmp_path)
    _,records = runner.source_anchors(template,root,"joint_v2_ssd1b")
    Path(records[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed since preflight"):
        runner.supplied_anchor(tmp_path/"new",dict(anchor_swap=dict(source_anchors=records,video_system="joint_v2")),item)


def test_native_video_uses_supplied_dreamlite_image_and_native_text_stack(monkeypatch):
    module = ModuleType("neodragon.utils.generation_utils")
    module.DEFAULT_NEGATIVE_PROMPT="negative"; module.DEFAULT_PROMPT_MODIFIER=", suffix"
    module.generate=lambda **kw: kw
    monkeypatch.setitem(sys.modules,"neodragon.utils.generation_utils",module)
    class Pipeline:
        mode="hybrid"; dit="dit"; vae="vae"; scheduler="scheduler"
        text_encoder_bundle="native text"; context_adapter="native adapter"
        config=SimpleNamespace(frames_per_unit=1,stages=[1,2,4],gen_confs=dict(hybrid=dict(
            num_inference_steps=[1,1,1],video_num_inference_steps=[1,1,1],do_classifier_free_guidance=False)))
        def __call__(self,**kw): raise AssertionError("Must not use Hybrid __call__ and regenerate SSD")
    image=Image.new("RGB",(32,16),"red")
    result=runner.generation.generate_frames(Pipeline(),None,None,"panda",image,"native_hybrid",torch.device("cpu"),torch.float32)
    assert result["image"] is image and result["text_encoder_bundle"] == "native text"
    assert result["context_adapter"] == "native adapter"
    assert result["video_num_inference_steps"] == [1,1,1] and not result["do_classifier_free_guidance"]


def test_swap_prepare_keeps_one_video_stack_and_both_export_variants(tmp_path,monkeypatch):
    root,template,baseline,parent,item = fixture_baseline(tmp_path)
    def asset_prepare(args):
        assert args.systems == list(protocol.SYSTEMS)
        protocol.atomic_json(args.output_dir/"vbench_info.json",[dict(prompt_en="panda",dimension=["scene"])])
        protocol.atomic_json(args.output_dir/"sample_plan.json",dict(samples=[item]))
        return template
    monkeypatch.setattr(runner.core,"prepare",asset_prepare)
    for case,cell in runner.SWAPS.items():
        args=SimpleNamespace(output_dir=tmp_path/case,baseline_run=root,swap=case)
        result=runner.prepare(args)
        assert runner.prepare(args) == result
        assert result["selected_systems"] == [cell["video"]]
        for variant in runner.dual.VARIANTS:
            contract=protocol.read_json(args.output_dir/variant/"run_contract.json")
            assert contract["anchor_swap"]["image_regeneration"] is False
            assert contract["anchor_swap"]["image_system"] == cell["image"]
            assert contract["selected_systems"] == [cell["video"]]


def test_baseline_effect_report_contains_all_dimensions(tmp_path):
    old=dict(quality_score=.8,semantic_score=.6,total_score=.76,raw={d:.5 for d in protocol.DIMENSIONS})
    new=deepcopy(old); new["total_score"]+=.01
    baseline=tmp_path/"baseline"; output=tmp_path/"swap"
    scores={v:{"joint_v2":old} for v in runner.dual.VARIANTS}
    protocol.atomic_json(baseline/"quicksr_ablation.json",dict(scores=scores))
    protocol.atomic_json(output/"quicksr_ablation.json",dict(scores={v:{"joint_v2":new} for v in runner.dual.VARIANTS}))
    for root in (baseline,output): protocol.atomic_json(root/"score_assets.json",[])
    protocol.atomic_json(output/"quicksr/run_contract.json",dict(scope="only image source changed",
        anchor_swap=dict(case="joint_v2_ssd1b",baseline_run=str(baseline),video_system="joint_v2",image_label="SSD1B")))
    runner.anchor_effect(SimpleNamespace(output_dir=output))
    report=protocol.read_json(output/"anchor_effect.json")
    assert report["rows"]["total_score"]["quicksr"]["change_pp"] == pytest.approx(1)
    assert len(report["rows"]) == 19


def test_job_pair_requests_two_independent_eight_gpu_jobs():
    for name,case in [("vbench_joint_v2_ssd1b_anchor_1node8gpu.sbatch","joint_v2_ssd1b"),
                      ("vbench_native_dreamlite_v11_anchor_1node8gpu.sbatch","native_hybrid_dreamlite_v11")]:
        path=ROOT/"scripts"/name
        subprocess.run(["bash","-n",str(path)],check=True)
        assert "#SBATCH --gres=gpu:8" in path.read_text() and f"SWAP_CASE={case}" in path.read_text()
    launcher=ROOT/"scripts/run_vbench_first_frame_swap.sh"
    subprocess.run(["bash","-n",str(launcher)],check=True)
    text=launcher.read_text()
    assert "SAMPLES_PER_PROMPT:-5" in text and "82.85" in text and "83.35" in text


@pytest.mark.parametrize("case", list(runner.SWAPS))
def test_defaults_are_full_five_sample_eight_gpu_cross_jobs(case):
    args = runner.parse_args(["--swap", case])
    assert args.workers == 8 and args.samples_per_prompt == 5
    assert args.max_prompts == 0 and args.seed_mapping == "extendable"
    assert args.output_dir.name == f"vbench_{case}_5x_dual"


@pytest.mark.parametrize("case", list(runner.SWAPS))
def test_score_worker_covers_both_variants_and_all_dimensions(tmp_path, monkeypatch, case):
    system = runner.SWAPS[case]["video"]
    protocol.atomic_json(tmp_path / "run_contract.json", dict(selected_systems=[system]))
    protocol.atomic_json(tmp_path / "quicksr/run_contract.json", dict(samples_per_prompt=5))
    commands = []
    monkeypatch.setattr(runner.dual.subprocess, "run", lambda argv, **kw: commands.append(argv))
    for rank in range(8):
        runner.dual.score_worker(SimpleNamespace(output_dir=tmp_path, vbench_repo=tmp_path, rank=rank, workers=8))
    tasks = {(Path(option(argv, "--videos")).parts[-3], int(option(argv, "--start"))) for argv in commands}
    assert len(commands) == len(tasks) == 32
    assert {variant for variant, _ in tasks} == set(runner.dual.VARIANTS)
    assert all(option(argv, "--expected-samples-per-prompt") == "5" for argv in commands)


@pytest.mark.parametrize("script,case", [
    ("vbench_joint_v2_ssd1b_anchor_1node8gpu.sbatch", "joint_v2_ssd1b"),
    ("vbench_native_dreamlite_v11_anchor_1node8gpu.sbatch", "native_hybrid_dreamlite_v11"),
])
def test_berzelius_wrapper_forwards_full_settings_and_restored_weight(launch, monkeypatch, script, case):
    monkeypatch.setattr(launch_helpers, "SCRIPT", ROOT / "scripts" / script)
    root, env, run = launch
    (root / "scripts/run_vbench_first_frame_swap.sh").write_text((ROOT / "scripts/run_vbench_first_frame_swap.sh").read_text())
    baseline = root / "output/vbench_joint_v2_vs_native_quicksr_berzelius_5x_dual"
    for name in ("generation_summary.json", "quicksr/run_contract.json", "quicksr_ablation.json", "score_assets.json"):
        protocol.atomic_json(baseline / name, {})
    weight = root / "output" / launch_helpers.JOINT_FILE
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"restored checkpoint")
    python = Path(env["CONDA_ENV"]) / "bin/python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
argv = sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output = pathlib.Path(argv[argv.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
(output / "anchor_effect.json").write_text("{}")
(output / "anchor_effect.md").write_text("mock anchor effect")
''')
    result, argv = run(BASELINE_RUN=str(baseline))
    assert result.returncode == 0, result.stderr
    assert "tools/benchmark_first_frame_swap.py" in argv
    assert option(argv, "--swap") == case
    assert option(argv, "--workers") == "8" and option(argv, "--samples-per-prompt") == "5"
    assert option(argv, "--seed-mapping") == "extendable" and option(argv, "--max-prompts") == "0"
    assert option(argv, "--joint-checkpoint") == "output/" + launch_helpers.JOINT_FILE
    assert option(argv, "--output-dir") == f"output/vbench_{case}_5x_dual"
    assert "mock anchor effect" in result.stdout
    assert not (root / "checkpoints/hf_mobile_ov" / launch_helpers.JOINT_FILE).exists()
    baseline.rename(root / "removed_baseline")
    Path(env["ARGV_LOG"]).unlink()
    result, argv = run(BASELINE_RUN=str(baseline))
    assert result.returncode != 0 and not argv
