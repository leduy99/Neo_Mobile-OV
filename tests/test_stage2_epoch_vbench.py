from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import benchmark_mobileov_stage2_epoch_dual as runner
from tools import vbench_joint_native_protocol as protocol
import test_berzelius_vbench_launch as launch_helpers
from test_berzelius_vbench_launch import launch, option
from test_stage2_media3epochs_dual_vbench import payload as small_payload

ROOT = Path(__file__).resolve().parents[1]


def payload(step=100000):
    value = small_payload()
    value["step"] = step
    value["contract"]["media_epochs"].update(steps_per_epoch=94659, steps=283977)
    return value


def reference_fixture(tmp_path):
    value = payload(283977)
    contract = dict(format="mobileov_stage2_media3epochs_vbench_dual_v1", system=runner.base.SYSTEM,
        checkpoint=dict(path="final.pt",sha256="f"*64,step=283977,final_step=283977),
        training_complete=True,max_prompts=0,samples_per_prompt=5,
        training_contract=value["contract"],connector_spec=value["connector_spec"],
        prompts=[f"prompt {i}" for i in range(944)],config_sha256="cfg",full_info_sha256="info",
        seed=2026,seed_mapping="extendable",seed_policy="unchanged independent per-video seed",
        exports={v:dict(width=512,height=320,frames=49,fps=24) for v in runner.base.VARIANTS},
        quicksr=dict(path="/old/cache/qsr.pt",sha256="qsr",variant="medium",scale=2,dtype="fp16",batch_frames=4),
        first_steps_per_stage=20,video_steps_per_stage=10,guidance=7.,video_guidance=5.,
        external_anchor=False,native_conditioner=False,conditioning="literal prompt; empty negative",
        dtype="bf16_compute_fp32_connector",library_versions=dict(torch="test"),
        vbench_source_commit="vbench",implementation_sha256=dict(original="unchanged"),scope="final epoch 3")
    assets = [dict(path="/old/cache/metric.pt",sha256="metric")]
    score = dict(quality_score=.8,semantic_score=.6,total_score=.76,raw={d:.5 for d in protocol.DIMENSIONS})
    scores = {v:{runner.base.SYSTEM:deepcopy(score)} for v in runner.base.VARIANTS}
    report = dict(status="complete",protocol=contract,evaluation_assets=assets,scores=scores)
    root = tmp_path/"final"
    for name, data in (("run_contract.json",contract),("quicksr_ablation.json",report),("score_assets.json",assets)):
        protocol.atomic_json(root/name,data)
    return root,contract,report,assets


@pytest.mark.parametrize("epoch,step",[(1,100000),(2,190000)])
def test_pinned_intermediate_reader_and_default_final_guard_both_work(tmp_path,epoch,step):
    value=payload(step); path=tmp_path/"weights.pt"; torch.save(value,path)
    original=runner.base.read_checkpoint
    with pytest.raises(ValueError,match="NOT complete"):
        original(path)
    with runner.checkpoint_policy(step,epoch,protocol.fingerprint(value["contract"])):
        loaded,_,target=runner.base.read_checkpoint(path)
        assert loaded["step"]==step and target==283977
    assert runner.base.read_checkpoint is original
    with pytest.raises(ValueError,match="NOT complete"):
        original(path)


@pytest.mark.parametrize("expected,epoch,match",[(190000,1,"Wrong checkpoint"),(100000,2,"not after epoch")])
def test_wrong_step_or_epoch_rejected_and_original_reader_restored(tmp_path,expected,epoch,match):
    path=tmp_path/"weights.pt"; torch.save(payload(),path)
    original=runner.base.read_checkpoint
    with pytest.raises(ValueError,match=match),runner.checkpoint_policy(expected,epoch):
        runner.base.read_checkpoint(path)
    assert runner.base.read_checkpoint is original


def test_wrong_training_run_or_corrupt_weights_cannot_be_admitted(tmp_path):
    value=payload(); path=tmp_path/"weights.pt"; torch.save(value,path)
    with pytest.raises(ValueError,match="different training contract"),runner.checkpoint_policy(100000,1,"other"):
        runner.base.read_checkpoint(path)
    value["dit"]["weight"][0,0]=float("nan"); torch.save(value,path)
    with pytest.raises(ValueError),runner.checkpoint_policy(100000,1):
        runner.base.read_checkpoint(path)


@pytest.mark.parametrize("epoch,step",[(1,100000),(2,190000)])
def test_cli_defaults_are_archived_weights_eight_gpus_and_five_samples(epoch,step):
    args=runner.parse_args(["--epoch",str(epoch),"--expected-step",str(step)])
    assert args.workers==8 and args.samples_per_prompt==5 and args.max_prompts==0
    assert args.checkpoint.name==f"stage2_dit_connector_step{step:06d}.pt"
    assert "/17641364/" in str(args.checkpoint)
    assert args.output_dir.name==f"vbench_mobileov_stage2_media3epochs_epoch{epoch}_step{step}_5x_dual"
    assert not args.allow_incomplete_checkpoint


@pytest.mark.parametrize("flags",[[],["--epoch","1"],["--expected-step","100000"],
    ["--epoch","1","--expected-step","-1"],
    ["--epoch","1","--expected-step","100000","--samples-per-prompt","1"],
    ["--epoch","1","--expected-step","100000","--allow-incomplete-checkpoint"]])
def test_cli_does_not_offer_a_generic_full_checkpoint_bypass(flags):
    with pytest.raises(SystemExit): runner.parse_args(flags)


def test_cli_help_includes_the_new_checkpoint_policy():
    result=subprocess.run([sys.executable,str(ROOT/"tools/benchmark_mobileov_stage2_epoch_dual.py"),"--help"],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "--expected-step" in result.stdout and "--reference-run" in result.stdout


@pytest.mark.parametrize("field",["seed","prompts","exports","guidance","library_versions", "implementation_sha256", "training_contract"])
def test_protocol_changes_from_final_epoch_are_rejected(tmp_path,field):
    _,reference,_,_=reference_fixture(tmp_path)
    current=deepcopy(reference); current[field]="changed"
    with pytest.raises(ValueError): runner.compare_protocol(reference,current)


def test_relocated_cache_is_allowed_only_for_identical_qsr_weights(tmp_path):
    _,reference,_,_=reference_fixture(tmp_path)
    current=deepcopy(reference); current["quicksr"]["path"]="/new/qsr.pt"
    runner.compare_protocol(reference,current)
    current["quicksr"]["sha256"]="changed"
    with pytest.raises(ValueError,match="QuickSR"): runner.compare_protocol(reference,current)


def test_final_reference_requires_complete_consistent_scores(tmp_path):
    root,contract,report,_=reference_fixture(tmp_path)
    assert runner.read_reference(root)[0]==contract
    report["protocol"]=dict(contract,seed=999)
    protocol.atomic_json(root/"quicksr_ablation.json",report)
    with pytest.raises(ValueError,match="do not match"):runner.read_reference(root)
    (root/"quicksr_ablation.json").unlink()
    with pytest.raises(FileNotFoundError,match="Missing completed"):runner.read_reference(root)


@pytest.mark.parametrize("epoch,step",[(1,100000),(2,190000)])
def test_prepare_keeps_intermediate_identity_and_reuses_final_reference(tmp_path,monkeypatch,epoch,step):
    reference,original,_,_=reference_fixture(tmp_path)
    reference_sha=protocol.sha256(reference/"run_contract.json")
    value=deepcopy(original)
    value.update(training_complete=False,checkpoint=dict(original["checkpoint"],path="intermediate.pt",step=step))
    def prepare(args):
        protocol.atomic_json(args.output_dir/"vbench_info.json",[])
        protocol.atomic_json(args.output_dir/"sample_plan.json",dict(samples=list(runner.base.items_for(value))))
        protocol.atomic_json(args.output_dir/"checkpoint_audit.json",dict(training_complete=False))
        return deepcopy(value)
    monkeypatch.setattr(runner.base,"prepare",prepare)
    args=SimpleNamespace(output_dir=tmp_path/"middle",reference_run=reference,max_prompts=0,
                         epoch=epoch,expected_step=step)
    result=runner.prepare(args)
    assert runner.prepare(args)==result
    assert result["training_complete"] is False
    policy=result["checkpoint_evaluation"]
    assert policy["expected_step"]==step and policy["media_epoch_progress"]==pytest.approx(step/94659)
    assert policy["reference_verified"] is True
    assert protocol.sha256(reference/"run_contract.json")==reference_sha
    assert len(protocol.read_json(args.output_dir/"sample_plan.json")["samples"])==4720
    with pytest.raises(ValueError,match="separate output"):
        runner.prepare(SimpleNamespace(**{**vars(args),"output_dir":reference}))


def test_worker_delegates_numerical_inference_under_pinned_policy(tmp_path,monkeypatch):
    policy=dict(expected_step=190000,requested_epoch=2,training_contract_sha256=protocol.fingerprint(payload(190000)["contract"]))
    protocol.atomic_json(tmp_path/"run_contract.json",dict(checkpoint_evaluation=policy))
    path=tmp_path/"weights.pt";torch.save(payload(190000),path)
    calls=[]
    def worker(args):
        loaded,_,_=runner.base.read_checkpoint(path)
        calls.append(loaded["step"])
    monkeypatch.setattr(runner.base,"generation_worker",worker)
    original=runner.base.read_checkpoint
    runner.generation_worker(SimpleNamespace(output_dir=tmp_path))
    assert calls==[190000] and runner.base.read_checkpoint is original


def prepared_contract(tmp_path):
    reference,contract,report,assets=reference_fixture(tmp_path)
    value=deepcopy(contract);value.update(training_complete=False,checkpoint=dict(contract["checkpoint"],step=100000))
    value["checkpoint_evaluation"]=dict(kind="intermediate_checkpoint",expected_step=100000,requested_epoch=1,
        media_epoch_progress=100000/94659,reference_run=str(reference),reference_verified=True,
        reference_contract_sha256=protocol.fingerprint(contract),training_contract_sha256=protocol.fingerprint(contract["training_contract"]))
    return value,report,assets


def test_assets_and_changed_final_reference_are_detected(tmp_path):
    contract,_,assets=prepared_contract(tmp_path)
    protocol.atomic_json(tmp_path/"score_assets.json",[dict(assets[0],path="/new/location")])
    runner.verify_score_assets(tmp_path,contract)
    protocol.atomic_json(tmp_path/"score_assets.json",[dict(assets[0],sha256="changed")])
    with pytest.raises(ValueError,match="evaluation weights"):runner.verify_score_assets(tmp_path,contract)
    contract["checkpoint_evaluation"]["reference_contract_sha256"]="changed"
    with pytest.raises(ValueError,match="changed after preflight"):runner.verified_reference(contract)


def test_full_intermediate_summary_keeps_training_incomplete_and_writes_all_dimensions(tmp_path,monkeypatch):
    contract,reference,assets=prepared_contract(tmp_path)
    output=tmp_path/"middle"
    protocol.atomic_json(output/"run_contract.json",contract)
    protocol.atomic_json(output/"score_assets.json",assets)
    score=deepcopy(reference["scores"]["quicksr"][runner.base.SYSTEM]);score["total_score"]+=.01
    module=ModuleType("evaluate_vbench_resume");module.tabulate=lambda _:deepcopy(score)
    monkeypatch.setitem(sys.modules,"evaluate_vbench_resume",module)
    monkeypatch.setattr(runner.base,"verify_generation",lambda _:None)
    monkeypatch.setattr(runner.subprocess,"run",lambda *args,**kw:None)
    for variant in runner.base.VARIANTS:
        for dimension in protocol.DIMENSIONS:
            protocol.atomic_json(output/variant/"scores"/dimension/"vbench"/f"{dimension}_eval_results.json",{dimension:[.5,[]]})
    runner.summarize(SimpleNamespace(output_dir=output))
    report=protocol.read_json(output/"quicksr_ablation.json")
    assert report["status"]=="complete" and report["protocol"]["training_complete"] is False
    comparison=protocol.read_json(output/"epoch_comparison.json")
    assert len(comparison["rows"])==19
    assert comparison["rows"]["total_score"]["quicksr"]["difference_pp"]==pytest.approx(1.)
    assert "1.056" in (output/"quicksr_ablation.md").read_text()
    contract["max_prompts"]=2;protocol.atomic_json(output/"run_contract.json",contract)
    with pytest.raises(ValueError,match="No full score"):runner.summarize(SimpleNamespace(output_dir=output))


def test_full_orchestration_warms_assets_then_generates_scores_and_summarizes(tmp_path,monkeypatch):
    contract,_,_=prepared_contract(tmp_path)
    args=SimpleNamespace(phase="run",output_dir=tmp_path/"middle",workers=8,max_prompts=0,
                         vbench_python=sys.executable,vbench_repo=tmp_path)
    monkeypatch.setattr(runner,"parse_args",lambda:args)
    monkeypatch.setattr(runner,"prepare",lambda _:contract)
    monkeypatch.setattr(runner.base.core,"gpu_tokens",lambda _:list(map(str,range(8))))
    events=[]
    monkeypatch.setattr(runner,"verify_score_assets",lambda *parts:events.append("assets"))
    monkeypatch.setattr(runner.base,"verify_generation",lambda _:events.append("verify_generation"))
    def process(argv,**kwargs):
        phase=option(argv,"--phase")
        assert phase in ("warm-scores","summarize")
        events.append(phase)
    def workers(commands,devices,folder,**kwargs):
        assert len(commands)==len(devices)==8
        phases={option(argv,"--phase") for argv in commands}
        assert len(phases)==1
        assert {option(argv,"--rank") for argv in commands}==set(map(str,range(8)))
        assert all("tools/benchmark_mobileov_stage2_epoch_dual.py" in argv[1] for argv in commands)
        events.append(next(iter(phases)))
    monkeypatch.setattr(runner.subprocess,"run",process)
    monkeypatch.setattr(runner.base.core,"run_processes",workers)
    runner.main()
    assert events==["warm-scores","assets","generation-worker","verify_generation","score-worker","summarize"]


@pytest.mark.parametrize("epoch,step",[(1,100000),(2,190000)])
def test_berzelius_epoch_jobs_pin_checkpoint_and_preserve_final(launch,monkeypatch,epoch,step):
    script=ROOT/f"scripts/vbench_mobileov_stage2_epoch{epoch}_1node8gpu.sbatch"
    monkeypatch.setattr(launch_helpers,"SCRIPT",script)
    root,env,run=launch
    launcher=ROOT/"scripts/run_vbench_mobileov_stage2_epoch.sh"
    (root/"scripts/run_vbench_mobileov_stage2_epoch.sh").write_text(launcher.read_text())
    weight=root/f"output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_step{step}.pt"
    weight.parent.mkdir(parents=True);weight.write_bytes(b"archived weights")
    reference=root/"output/vbench_mobileov_stage2_media3epochs_5x_dual"
    for name in ("run_contract.json","quicksr_ablation.json","score_assets.json"):
        protocol.atomic_json(reference/name,{})
    python=Path(env["CONDA_ENV"])/"bin/python"
    python.write_text(f"#!{sys.executable}\n"+'''import json,os,pathlib,sys
argv=sys.argv[1:]
pathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(argv))
output=pathlib.Path(argv[argv.index("--output-dir")+1]);output.mkdir(parents=True,exist_ok=True)
(output/"epoch_comparison.json").write_text("{}")
(output/"epoch_comparison.md").write_text("mock epoch comparison")
''')
    result,argv=run()
    assert result.returncode==0,result.stderr
    assert "tools/benchmark_mobileov_stage2_epoch_dual.py" in argv
    assert option(argv,"--epoch")==str(epoch) and option(argv,"--expected-step")==str(step)
    assert option(argv,"--checkpoint")==str(weight.relative_to(root))
    assert option(argv,"--workers")=="8" and option(argv,"--samples-per-prompt")=="5"
    assert option(argv,"--seed")=="2026"
    assert "--allow-incomplete-checkpoint" not in argv
    assert f"epoch{epoch}_step{step}_5x_dual" in option(argv,"--output-dir")
    assert "mock epoch comparison" in result.stdout
    Path(env["ARGV_LOG"]).unlink();weight.unlink()
    result,argv=run()
    assert result.returncode!=0 and not argv and "no latest/HF fallback" in result.stderr
    for path in (script,launcher):subprocess.run(["bash","-n",str(path)],check=True)
    assert "#SBATCH --gres=gpu:8" in script.read_text()
