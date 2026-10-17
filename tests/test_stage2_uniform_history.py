from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training.distributed import DistributedContext
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from new_mobile_ov.training.stage2_history import corrupt_stage_history, summarize_history, MAX_HISTORY_SIGMA
from tools import train_mobileov_stage1 as trainer
from tools import plan_mobileov_stage2_history as planner
from tools.data_prepare.download_alignment_images import sha256_file
from test_stage2_training import TinyDiT, scheduler
from test_stage2_first_frame_focus import parent_metadata, assert_same

ROOT = Path(__file__).resolve().parents[1]


def cli(parent="parent.pt", output="unused", expected=283977):
    return ["--phase", "2", "--init-stage2", str(parent), "--output-dir", str(output),
            "--expected-stage2-step", str(expected), "--media-epochs", "1", "--media-start-epoch", "3",
            "--history-corrupt-max", str(MAX_HISTORY_SIGMA), "--reconstruction-weight", ".1"]


def test_new_recipe_is_explicit_uniform_and_cannot_silently_change_old_forks():
    args = trainer.parse_args(cli())
    assert args.t2v_first_probability is None and args.history_corrupt_max == 1/3
    assert args.reconstruction_weight == .1 and args.lr == 2e-5
    for extra in (["--reconstruction-weight", ".2"], ["--t2v-first-probability", ".5"],
                  ["--history-corrupt-max", "0"], ["--history-corrupt-max", "nan"],
                  ["--history-corrupt-max", ".34"], ["--media-start-epoch", "0"]):
        with pytest.raises(SystemExit): trainer.parse_args(cli() + extra)
    with pytest.raises(SystemExit):
        trainer.parse_args(["--output-dir", "x", "--history-corrupt-max", ".1"])
    old = trainer.parse_args(["--phase", "2", "--init-stage2", "old.pt", "--output-dir", "x",
                              "--media-epochs", "1", "--media-start-epoch", "3",
                              "--t2v-first-probability", ".5", "--reconstruction-weight", ".2"])
    assert old.history_corrupt_max == 0


def test_stage_resolution_corruption_preserves_grouped_anchor_and_inputs():
    history = [torch.ones(2, 4, 3, 4, 6), torch.ones(2, 4, 1, 8, 12)]
    before = [value.clone() for value in history]
    stats = {}
    noisy = corrupt_stage_history(history, maximum=1/3, generator=torch.Generator().manual_seed(42),
                                  preserve_first_anchor=True, stats=stats)
    assert torch.equal(noisy[0][:, :, :1], history[0][:, :, :1])
    assert not torch.equal(noisy[0][:, :, 1:], history[0][:, :, 1:])
    assert not torch.equal(noisy[1], history[1])
    assert all(torch.equal(a, b) for a, b in zip(history, before))
    assert stats["eligible_units"] == stats["applied_units"] == 6
    assert stats["protected_units"] == 2 and stats["change_squared_sum"] > 0
    report = summarize_history([stats], DistributedContext(0, 0, 1, torch.device("cpu")), 1/3)
    assert 0 < report["sigma_mean"] < 1/3 and report["changed_rms"] > 0


def test_noise_is_added_at_each_prepared_resolution_without_downsample_attenuation():
    history = [torch.zeros(32, 4, 2, 8, 12), torch.zeros(32, 4, 1, 16, 24)]
    sigma = torch.rand(32, generator=torch.Generator().manual_seed(7)) / 3
    noisy = corrupt_stage_history(history, maximum=1/3, generator=torch.Generator().manual_seed(7))
    for value in noisy:
        recovered_noise = value / sigma.reshape(32, 1, 1, 1, 1)
        assert recovered_noise.std().item() == pytest.approx(1., abs=.03)


def test_empty_or_observed_only_history_does_not_consume_noise_rng():
    rng = torch.Generator().manual_seed(5)
    state = rng.get_state()
    for history, protected in (([], False), ([torch.ones(1, 4, 1, 4, 6)], True)):
        stats = {}
        result = corrupt_stage_history(history, maximum=1/3, generator=rng, preserve_first_anchor=protected, stats=stats)
        assert result is history and stats["eligible_units"] == 0
        assert torch.equal(state, rng.get_state())
    with pytest.raises(ValueError, match="independent"):
        corrupt_stage_history([torch.ones(1, 4, 1, 4, 6)], maximum=1/3, generator=None)
    with pytest.raises(RuntimeError, match="actual forward"):
        summarize_history([{}], DistributedContext(0, 0, 1, torch.device("cpu")), 1/3)


class HistoryDiT(TinyDiT):
    def forward(self, sample, **kwargs):
        result = super().forward(sample, **kwargs)[0]
        signal = sum(value.float().mean() for value in sample[0][:-1])
        return [result + self.weight * signal * .1]


def test_actual_joint_owner_receives_changed_history_but_identical_current_flow(monkeypatch):
    prepared, seen = [], []
    def prepare(past, stages, cfg):
        # The history is spatially resized BEFORE the corruption hook.
        group = torch.cat(past, dim=2)
        levels = [[torch.nn.functional.interpolate(group, size=(group.shape[2], 4, 6), mode="trilinear")]] * stages
        prepared.append(levels[0][0].clone())
        return levels
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(_prepare_past_condition_latents=prepare))
    class Capture(HistoryDiT):
        def forward(self, sample, **kwargs):
            seen.append([value.detach().clone() for value in sample[0]])
            return super().forward(sample, **kwargs)
    torch.manual_seed(3)
    owner = stage2.JointFlowModel(Stage1Connector(8, 16, 12), Capture())
    noisy = deepcopy(owner)
    noisy.history_corrupt_max = 1/3
    layers, mask = [torch.randn(1, 9, 8) for _ in range(3)], torch.ones(1, 9)
    latent = torch.randn(1, 4, 7, 16, 24)
    clean_before = latent.clone()
    a, b = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    first = owner(layers, mask, latent, scheduler(), stage=2, unit=6, generator=a)
    stats = {}
    second = noisy(layers, mask, latent, scheduler(), stage=2, unit=6, generator=b,
                   history_generator=torch.Generator().manual_seed(42), preserve_first_anchor=True, history_stats=stats)
    second.backward()
    assert torch.equal(a.get_state(), b.get_state()) and torch.equal(latent, clean_before)
    assert torch.equal(seen[0][-1], seen[1][-1])  # The current noisy latent and target RNG did not change.
    assert torch.equal(seen[1][0][:, :, :1], prepared[1][:, :, :1])
    assert not torch.equal(seen[1][0][:, :, 1:], prepared[1][:, :, 1:])
    assert first != second and stats["applied_units"] == 5
    assert stage2.gradient_norm(noisy.connector) > 0 and stage2.gradient_norm(noisy.dit) > 0


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_native_pyramid_mmdit_blocks_support_corrupted_history_on_cpu(stage, monkeypatch):
    repo=ROOT/"checkpoints/neodragon_repo"
    if not repo.is_dir(): pytest.skip("Optional native NeoDragon checkout is not installed")
    monkeypatch.syspath_prepend(str(repo))
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from test_stage2_training import batch
    torch.manual_seed(19)
    dit=PyramidMMDiT(sample_size=16,in_channels=4,num_layers=2,num_attention_heads=2,
        attention_head_dim=16,caption_projection_dim=32,pooled_projection_dim=12,
        pos_embed_max_size=32,use_gradient_checkpointing=True,gradient_checkpointing_ratio=0.)
    for module in dit.modules():
        if isinstance(module,torch.nn.Linear):torch.nn.init.normal_(module.weight,std=.03)
    owner=stage2.JointFlowModel(Stage1Connector(8,32,12),dit,history_corrupt_max=1/3).train()
    layers,mask,latent=batch(0,0,1)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        loss=owner([value.bfloat16() for value in layers],mask,latent.bfloat16(),
            PyramidFlowMatchEulerDiscreteScheduler(),stage=stage,unit=6,generator=torch.Generator().manual_seed(2),
            history_generator=torch.Generator().manual_seed(42),preserve_first_anchor=True)
    loss.backward()
    assert torch.isfinite(loss) and owner.last_history_stats["applied_units"]==5
    assert owner.last_history_stats["change_squared_sum"]>0
    assert stage2.gradient_norm(dit)>0 and stage2.gradient_norm(owner.connector)>0


def test_real_trainer_history_fork_resume_is_exact(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage1_summary.json").write_text("{}")
    config, smol = tmp_path / "config", tmp_path / "smol"
    config.write_text("config"); smol.write_text("smol")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(smol)))
    plan = media.MediaEpochPlan(3, 9, 1, 2, epochs=1, start_epoch=3)
    class StochasticDiT(HistoryDiT):
        def forward(self, *args, **kwargs):
            value = super().forward(*args, **kwargs)[0]
            return [torch.nn.functional.dropout(value, .2, training=self.training)]
    def stack(args, device):
        encoder, vae, dit = torch.nn.Linear(2,2), torch.nn.Linear(2,2), StochasticDiT().bfloat16()
        for module in (encoder, vae, dit): module.eval().requires_grad_(False)
        encoder.processor_sha256 = "processor"
        return cfg, encoder, Stage1Connector(8,16,12), dit, vae, scheduler()
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, *, plan, rank, start_step, **kwargs):
            self.plan, self.rank, self.start = plan, rank, start_step
            self.records = {"t2i":range(plan.images), "t2v":range(plan.videos)}
        def __len__(self): return (self.plan.steps-self.start)*self.plan.accumulation
        def __getitem__(self, index):
            step, micro = divmod(index, self.plan.accumulation)
            slot = self.plan.slot(step+self.start, micro, self.rank)
            return dict(task=slot["source"], sample_id=f"{slot['source']}:{slot['index']}", **slot)
    def encode(encoder, vae, sample, device, rng, dropout, **kwargs):
        return ([torch.randn(1,9,8,generator=rng) for _ in range(3)], torch.ones(1,9),
                torch.randn(1,4,7 if sample["task"]=="t2v" else 1,16,24,generator=rng), False)
    def reconstruction(encoder, sample):
        rng=torch.Generator().manual_seed(sample["micro"]+1)
        return [torch.randn(1,13,8,generator=rng) for _ in range(3)], torch.ones(1,13)
    monkeypatch.setattr(trainer,"load_stack",stack)
    monkeypatch.setattr(trainer,"verify_release",lambda _:None)
    monkeypatch.setattr(media,"plan_from_release",lambda *a,**kw:plan)
    monkeypatch.setattr(media,"MediaEpochDataset",Dataset)
    monkeypatch.setattr(trainer,"encode_sample",encode)
    monkeypatch.setattr(trainer,"validate",lambda *a:{"diagnostic":float(torch.randn(()))})
    monkeypatch.setattr(stage2,"reconstruction_condition",reconstruction)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda _:0)
    torch.manual_seed(plan.seed)
    _, encoder, connector, dit, vae, _=stack(None,None)
    parent=parent_metadata(plan,sha256_file(data/"stage1_summary.json"))
    parent["contract"].update(config_sha256=sha256_file(config),processor="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        processor_sha256="processor",smolvlm2_sha256=sha256_file(smol),
        frozen_weights_sha256=frozen_signatures(dict(smolvlm2=encoder,vae=vae)),short_side=320,long_side=512,
        max_tokens=2048,condition_dropout=.1,vae_window_size=8,geometry="nearest_aspect_bucket_resize_full_view_no_crop",
        temporal="49_distinct_frames_over_whole_source_clip; normalized_duration_not_native_fps")
    parent.update(connector_spec=connector.spec,connector=deepcopy(connector.state_dict()),dit={"weight":torch.tensor(1.00012345)})
    init=tmp_path/"parent.pt";torch.save(parent,init)
    ctx=DistributedContext(0,0,1,torch.device("cpu"))
    def run(name,extra=()):
        out=tmp_path/name;out.mkdir(exist_ok=True)
        args=trainer.parse_args([*cli(init,out,parent["step"]),"--config",str(config),"--data-root",str(data),
            "--workers","0","--log-every","1","--save-every","4","--archive-every","4",
            "--validate-every","4","--verify-frozen",*extra])
        trainer.train(args,ctx)
        return out
    full=run("full")
    paused=run("paused",["--stop-after",str(parent["step"]+2)])
    run("paused",["--resume",str(paused/"stage2_resume.pt")])
    a=torch.load(full/"stage2_resume.pt",weights_only=True)
    b=torch.load(paused/"stage2_resume.pt",weights_only=True)
    assert_same(a,b)
    assert "first_frame_focus" not in a["contract"] and a["contract"]["auxiliary_reconstruction"]["weight"]==.1
    assert a["contract"]["history_corruption"]["max_sigma"]==1/3
    assert (full/"history_corruption_audit.json").is_file()
    rows=[json.loads(line) for line in (full/"history.jsonl").read_text().splitlines()]
    assert any(row["history_corruption"]["changed_rms"] for row in rows)
    old=deepcopy(a);old["step"]-=1;old.pop("rng_states")
    with pytest.raises(ValueError,match="RNG"):trainer.validate_resume(old,a["contract"],a["connector_spec"])


def ddp_history_worker(rank, directory):
    torch.set_num_threads(1)
    sys.modules["neodragon.utils.generation_utils"] = SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg:[past]*stages)
    dist.init_process_group("gloo",init_method=f"file://{directory}/init",rank=rank,world_size=2)
    try:
        torch.manual_seed(42)
        owner=stage2.JointFlowModel(Stage1Connector(8,16,12),HistoryDiT(),history_corrupt_max=1/3)
        model=torch.nn.parallel.DistributedDataParallel(owner,find_unused_parameters=True,broadcast_buffers=False)
        opt=torch.optim.AdamW(owner.parameters(),lr=2e-5)
        opt.zero_grad()
        records=[]
        from contextlib import nullcontext
        from test_stage2_training import batch
        for micro in range(2):
            values=batch(rank,0,micro)
            with model.no_sync() if micro==0 else nullcontext():
                loss=model(*values,scheduler(),stage=(rank+micro)%3,unit=0 if micro==0 else rank+2,
                    generator=torch.Generator().manual_seed(500+rank*2+micro),
                    history_generator=torch.Generator().manual_seed(800+rank*2+micro),preserve_first_anchor=rank==1)
                records.append(owner.last_history_stats)
                (loss/2).backward()
        report=summarize_history(records,DistributedContext(rank,rank,2,torch.device("cpu")),1/3)
        opt.step()
        torch.save(dict(state=owner.state_dict(),report=report),Path(directory)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_history_forward_audit_and_gradients_match_global_batch(tmp_path,monkeypatch):
    mp.spawn(ddp_history_worker,args=(str(tmp_path),),nprocs=2,join=True)
    monkeypatch.setitem(sys.modules,"neodragon.utils.generation_utils",SimpleNamespace(
        _prepare_past_condition_latents=lambda past,stages,cfg:[past]*stages))
    from test_stage2_training import batch
    torch.manual_seed(42)
    owner=stage2.JointFlowModel(Stage1Connector(8,16,12),HistoryDiT(),history_corrupt_max=1/3)
    opt=torch.optim.AdamW(owner.parameters(),lr=2e-5)
    opt.zero_grad()
    for rank in range(2):
        for micro in range(2):
            loss=owner(*batch(rank,0,micro),scheduler(),stage=(rank+micro)%3,unit=0 if micro==0 else rank+2,
                generator=torch.Generator().manual_seed(500+rank*2+micro),
                history_generator=torch.Generator().manual_seed(800+rank*2+micro),preserve_first_anchor=rank==1)
            (loss/4).backward()
    opt.step()
    a=torch.load(tmp_path/"rank0.pt",weights_only=True)
    b=torch.load(tmp_path/"rank1.pt",weights_only=True)
    assert a["report"]==b["report"]
    assert a["report"]["applied_units"]==a["report"]["eligible_units"]==4
    assert a["report"]["preserved_anchor_units"]==1 and a["report"]["changed_rms"]>0
    for key,value in owner.state_dict().items():
        torch.testing.assert_close(a["state"][key],value,rtol=2e-5,atol=2e-6)
        assert torch.equal(a["state"][key],b["state"][key])


def test_budget_and_berzelius_launcher_keep_parent_uniform_recipe_and_continuation(tmp_path, monkeypatch):
    data=tmp_path/"data";data.mkdir();(data/"stage1_summary.json").write_text("{}")
    plan=media.MediaEpochPlan(3,5,2,2,epochs=1,start_epoch=3)
    parent=tmp_path/"parent.pt";torch.save(parent_metadata(plan,sha256_file(data/"stage1_summary.json")),parent)
    monkeypatch.setattr(planner.parent_plan,"plan_from_release",lambda *a,**kw:plan)
    args=SimpleNamespace(parent=parent,parent_step=3*plan.steps_per_epoch,parent_sha256=sha256_file(parent),
        data_root=data,world_size=2,accumulation=2,seed=plan.seed,min_train_videos=1,require_expanded_release=False,
        integration_smoke=False,output=tmp_path/"out/plan.json",audit_samples=0)
    report=planner.plan(args)
    assert report["loss_weights"]["reconstruction"]==.1 and report["t2v_first_probability"] is None
    assert report["history_noise"]==1/3 and report["additional_updates"]==plan.steps
    assert planner.plan(args)==report
    bin_dir=tmp_path/"bin";bin_dir.mkdir();out=tmp_path/"job";out.mkdir();calls=tmp_path/"calls"
    env=dict(os.environ,PROJECT_ROOT=str(ROOT),SLURM_JOB_ID="123",OUT=str(out),INIT_STAGE2=str(parent),DATA_ROOT=str(data),
        CONDA_ENV=str(tmp_path),PATH=f"{bin_dir}:{os.environ['PATH']}",CALLS=str(calls),TMPDIR=str(tmp_path/"tmp"))
    for key in ("STEPS","RESUME","INIT_CONNECTOR","AUTO_CONTINUE","T2V_FIRST_PROBABILITY"):env.pop(key,None)
    for name,body in {"module":"exit 0","python":'echo "PLAN $*" >> "$CALLS"',
        "torchrun":'echo "TRAIN $*" >> "$CALLS"; exit "${TRAIN_RC:-0}"',
        "sbatch":'echo "CONTINUE $* RESUME=$RESUME OUT=$OUT" >> "$CALLS"; echo 124'}.items():
        path=bin_dir/name;path.write_text("#!/bin/bash\n"+body+"\n");path.chmod(0o755)
    script=ROOT/"scripts/train_mobileov_stage2_uniform_history_1node8gpu.sbatch"
    def launch(**kw):return subprocess.run(["bash",str(script)],env={**env,**kw},capture_output=True,text=True)
    result=launch();assert result.returncode==0,result.stderr
    text=calls.read_text();assert "--history-corrupt-max 0.3333333333333333" in text
    assert "--reconstruction-weight .1" in text and "--t2v-first-probability" not in text
    (out/"training_paused.json").write_text("{}");(out/"stage2_resume.pt").write_text("resume")
    assert launch().returncode==0 and "--dependency=afterok:123 --export=ALL" in calls.read_text()
    calls.unlink()
    assert launch(TRAIN_RC="1").returncode!=0 and "CONTINUE" not in calls.read_text()
    assert launch(STEPS="1000").returncode!=0 and launch(T2V_FIRST_PROBABILITY=".5").returncode!=0
    for path in (script,ROOT/"scripts/smoke_mobileov_stage2_history_local.sbatch"):
        subprocess.run(["bash","-n",str(path)],check=True)
