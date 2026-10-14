from collections import Counter
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training import stage2_alignment as stage2
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from new_mobile_ov.training.distributed import DistributedContext
from tools import train_mobileov_stage1 as trainer
from tools import plan_mobileov_stage2_first_frame as planner
from tools.data_prepare.download_alignment_images import sha256_file
from test_stage2_training import TinyDiT, scheduler, install_history, batch


def fork_cli(parent="parent.pt", output="unused", expected=283977):
    return ["--phase", "2", "--init-stage2", str(parent), "--output-dir", str(output),
            "--expected-stage2-step", str(expected), "--media-epochs", "1", "--media-start-epoch", "3",
            "--t2v-first-probability", ".5", "--reconstruction-weight", ".2"]


def test_fork_cli_is_explicit_and_legacy_defaults_unchanged():
    args = trainer.parse_args(fork_cli())
    assert args.lr == 2e-5 and args.accumulation == 2 and args.media_start_epoch == 3
    for extra in (["--init-connector", "old.pt"], ["--media-start-epoch", "0"],
                  ["--reconstruction-weight", ".1"], ["--t2v-first-probability", "nan"],
                  ["--media-epochs", "3"], ["--steps", "10000"], ["--integration-smoke"],
                  ["--expected-stage2-sha256", "not_a_hash"]):
        with pytest.raises(SystemExit):
            trainer.parse_args([*fork_cli(), *extra])
    with pytest.raises(SystemExit):
        trainer.parse_args(["--phase", "2", "--init-connector", "old.pt", "--output-dir", "x",
                           "--t2v-first-probability", ".5"])
    with pytest.raises(SystemExit):
        trainer.parse_args([*fork_cli(), "--integration-smoke", "--stop-after", "284000"])


def test_fourth_pass_exact_offset_weights_counts_and_resume():
    full = media.MediaEpochPlan(3, 5, 2, 2, epochs=6)
    fork = media.MediaEpochPlan(3, 5, 2, 2, epochs=1, start_epoch=3)
    seen, weights = Counter(), Counter()
    for step in range(fork.steps):
        for micro in range(2):
            for rank in range(2):
                slot = fork.slot(step, micro, rank)
                assert slot == full.slot(step + 3 * fork.steps_per_epoch, micro, rank)
                assert slot["epoch"] == 3
                if not slot["padding"]:
                    seen[slot["source"], slot["index"]] += 1
                    weights[slot["source"]] += slot["loss_scale"] / fork.global_batch / fork.steps_per_epoch
    assert len(seen) == 8 and set(seen.values()) == {1}
    assert weights == pytest.approx({"t2i": .5, "t2v": .5})
    assert sum(fork.contract()["presentations_by_video_mode"].values()) == 5
    assert "per_video_mode_presentations" not in fork.contract()
    actual = media.MediaEpochPlan(984636, 529902, 8, 2, epochs=1, start_epoch=3)
    assert actual.steps == 94659 and actual.contract()["sampler_step_offset"] == 283977


def test_first_unit_bias_only_affects_t2v_and_preserves_future_coverage():
    rng = torch.Generator().manual_seed(42)
    values = Counter(media.sample_unit("t2v", 7, "cpu", rng, t2v_first_probability=.5) for _ in range(20000))
    assert .48 < values[0] / 20000 < .52
    assert set(values) == set(range(7))
    for mode in ("i2v_reference", "i2v_anchor", "t2i"):
        a, b = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
        assert [media.sample_unit(mode, 7, "cpu", a) for _ in range(100)] == [
            media.sample_unit(mode, 7, "cpu", b, t2v_first_probability=.5) for _ in range(100)]
    assert media.sample_unit("t2v", 1, "cpu", rng, t2v_first_probability=.5) == 0
    for value in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            media.sample_unit("t2v", 7, "cpu", rng, t2v_first_probability=value)


def parent_metadata(plan, summary_hash):
    previous = media.MediaEpochPlan(plan.images, plan.videos, plan.world_size, plan.accumulation, seed=plan.seed)
    return dict(format=stage2.FORMAT, step=previous.steps,
                contract=dict(steps=previous.steps, media_epochs=previous.contract(), data_summary_sha256=summary_hash))


def test_parent_rejects_partial_old_release_and_mismatched_epoch(tmp_path):
    plan = media.MediaEpochPlan(3, 5, 2, 2, epochs=1, start_epoch=3)
    parent = parent_metadata(plan, "summary")
    kwargs = dict(expected_step=parent["step"], data_summary_sha256="summary")
    assert media.validate_fork_parent(parent, plan, **kwargs) == parent["step"]
    for key, value in (("steps", 1), ("data_summary_sha256", "changed")):
        changed = copy.deepcopy(parent)
        changed["contract"][key] = value
        with pytest.raises(ValueError):
            media.validate_fork_parent(changed, plan, **kwargs)
    changed = copy.deepcopy(parent)
    changed["contract"]["media_epochs"]["videos"] = 1
    with pytest.raises(ValueError, match="counts"):
        media.validate_fork_parent(changed, plan, **kwargs)
    assert media.validate_fork_parent(changed, plan, integration_smoke=True, **kwargs) == parent["step"]


def test_fp32_parent_is_loaded_without_bf16_roundtrip():
    connector, dit = Stage1Connector(8, 16, 12), TinyDiT().bfloat16().requires_grad_(False)
    contract = dict(config_sha256="cfg", processor="proc", processor_sha256="hash",
                    frozen_weights_sha256={"smolvlm2": "smol", "vae": "vae"}, short_side=320,
                    long_side=512, max_tokens=2048, condition_dropout=.1, vae_window_size=8,
                    geometry="full", temporal="whole")
    parent = dict(format=stage2.FORMAT, step=12, connector_spec=connector.spec,
                  connector=copy.deepcopy(connector.state_dict()), dit={"weight": torch.tensor(1.00012345)},
                  contract=copy.deepcopy(contract))
    args = SimpleNamespace(expected_stage2_step=12, gradient_checkpointing=True)
    stage2.initialize_from_stage2(parent, connector, dit, contract=contract, args=args)
    assert dit.weight.dtype == torch.float32 and torch.equal(dit.weight, parent["dit"]["weight"])
    assert all(p.requires_grad for m in (connector, dit) for p in m.parameters())
    bad = copy.deepcopy(parent)
    bad["contract"]["processor_sha256"] = "other"
    with pytest.raises(ValueError, match="processor"):
        stage2.initialize_from_stage2(bad, connector, dit, contract=contract, args=args)


def focus_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    install_history()
    dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        owner = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
        wrapped = torch.nn.parallel.DistributedDataParallel(owner, find_unused_parameters=True,
                                                           gradient_as_bucket_view=True, broadcast_buffers=False)
        opt = torch.optim.AdamW(owner.parameters(), lr=2e-5)
        plan = media.MediaEpochPlan(3, 5, 2, 2, epochs=1, start_epoch=3)
        for step in range(plan.steps):
            opt.zero_grad()
            for micro in range(2):
                slot = plan.slot(step, micro, rank)
                values = batch(rank, step, int(slot["source"] == "t2v"))
                rng = trainer.generator_for(torch.device("cpu"), plan.seed, slot["micro"], rank)
                unit = media.sample_unit(slot["mode"], values[2].shape[2], "cpu", rng, t2v_first_probability=.5)
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    loss = wrapped(*values, scheduler(), stage=(step + rank) % 3, unit=unit, generator=rng)
                    (loss * slot["loss_scale"] / 2).backward()
                    if slot["source"] == "t2i":
                        rec = wrapped(*values, scheduler(), stage=2, unit=0, generator=rng)
                        (rec * .4 * slot["loss_scale"] / 2).backward()
            opt.step()
        torch.save(owner.state_dict(), Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_focused_ddp_accumulation_matches_global_batch(tmp_path, monkeypatch):
    mp.spawn(focus_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    torch.manual_seed(42)
    owner = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
    opt = torch.optim.AdamW(owner.parameters(), lr=2e-5)
    plan = media.MediaEpochPlan(3, 5, 2, 2, epochs=1, start_epoch=3)
    for step in range(plan.steps):
        opt.zero_grad()
        for rank in range(2):
            for micro in range(2):
                slot = plan.slot(step, micro, rank)
                values = batch(rank, step, int(slot["source"] == "t2v"))
                rng = trainer.generator_for(torch.device("cpu"), plan.seed, slot["micro"], rank)
                unit = media.sample_unit(slot["mode"], values[2].shape[2], "cpu", rng, t2v_first_probability=.5)
                loss = owner(*values, scheduler(), stage=(step + rank) % 3, unit=unit, generator=rng)
                (loss * slot["loss_scale"] / 4).backward()
                if slot["source"] == "t2i":
                    rec = owner(*values, scheduler(), stage=2, unit=0, generator=rng)
                    (rec * .4 * slot["loss_scale"] / 4).backward()
        opt.step()
    a = torch.load(tmp_path / "rank0.pt", weights_only=True)
    b = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, value in owner.state_dict().items():
        assert torch.equal(a[key], b[key])
        torch.testing.assert_close(a[key], value, atol=1e-6, rtol=1e-4)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def test_real_runner_fork_fresh_warmup_resume_rng_and_split_metrics(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage1_summary.json").write_text("{}")
    config, smol = tmp_path / "config", tmp_path / "smol"
    config.write_text("config")
    smol.write_text("smol")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(smol)))
    plan = media.MediaEpochPlan(3, 5, 1, 2, epochs=1, start_epoch=3)
    class StochasticDiT(TinyDiT):
        def forward(self, *args, **kwargs):
            value = super().forward(*args, **kwargs)[0]
            return [torch.nn.functional.dropout(value, .2, training=self.training)]
    def stack(args, device):
        encoder, vae, dit = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), StochasticDiT().bfloat16()
        for module in (encoder, vae, dit):
            module.eval().requires_grad_(False)
        encoder.processor_sha256 = "processor"
        return cfg, encoder, Stage1Connector(8, 16, 12), dit, vae, scheduler()
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, *, plan, rank, start_step, **kwargs):
            self.plan, self.rank, self.start = plan, rank, start_step
            self.records = {"t2i": range(plan.images), "t2v": range(plan.videos)}
        def __len__(self):
            return (self.plan.steps - self.start) * self.plan.accumulation
        def __getitem__(self, index):
            step, micro = divmod(index, self.plan.accumulation)
            slot = self.plan.slot(step + self.start, micro, self.rank)
            return dict(task=slot["source"], sample_id=f"{slot['source']}:{slot['index']}", **slot)
    def encode(encoder, vae, sample, device, rng, dropout, **kwargs):
        layers = [torch.randn(1, 9, 8, generator=rng) for _ in range(3)]
        latent = torch.randn(1, 4, 7 if sample["task"] == "t2v" else 1, 16, 24, generator=rng)
        return layers, torch.ones(1, 9), latent, False
    def reconstruction(encoder, sample):
        rng = torch.Generator().manual_seed(sample["micro"] + 1)
        return [torch.randn(1, 13, 8, generator=rng) for _ in range(3)], torch.ones(1, 13)
    monkeypatch.setattr(trainer, "load_stack", stack)
    monkeypatch.setattr(trainer, "verify_release", lambda root: None)
    monkeypatch.setattr(media, "plan_from_release", lambda *args, **kwargs: plan)
    monkeypatch.setattr(media, "MediaEpochDataset", Dataset)
    monkeypatch.setattr(trainer, "encode_sample", encode)
    monkeypatch.setattr(trainer, "validate", lambda *args: {"diagnostic": float(torch.randn(()))})
    monkeypatch.setattr(stage2, "reconstruction_condition", reconstruction)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    torch.manual_seed(plan.seed)
    _, encoder, connector, dit, vae, _ = stack(None, None)
    parent = parent_metadata(plan, sha256_file(data / "stage1_summary.json"))
    parent["contract"].update(config_sha256=sha256_file(config), processor="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        processor_sha256="processor", smolvlm2_sha256=sha256_file(smol),
        frozen_weights_sha256=frozen_signatures(dict(smolvlm2=encoder, vae=vae)),
        short_side=320, long_side=512, max_tokens=2048, condition_dropout=.1, vae_window_size=8,
        geometry="nearest_aspect_bucket_resize_full_view_no_crop",
        temporal="49_distinct_frames_over_whole_source_clip; normalized_duration_not_native_fps")
    parent.update(connector_spec=connector.spec, connector=copy.deepcopy(connector.state_dict()),
                  dit={"weight": torch.tensor(1.00012345)})
    init = tmp_path / "parent.pt"
    torch.save(parent, init)
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    def run(name, extra=()):
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        args = trainer.parse_args([*fork_cli(init, out, parent["step"]), "--config", str(config),
            "--data-root", str(data), "--workers", "0", "--log-every", "1", "--save-every", "4",
            "--archive-every", "4", "--validate-every", "4", "--verify-frozen", *extra])
        trainer.train(args, ctx)
        return out
    full = run("full")
    paused = run("paused", ["--stop-after", str(parent["step"] + 2)])
    paused_checkpoint = torch.load(paused / "stage2_resume.pt", weights_only=True)
    run("paused", ["--resume", str(paused / "stage2_resume.pt")])
    a = torch.load(full / "stage2_resume.pt", weights_only=True)
    b = torch.load(paused / "stage2_resume.pt", weights_only=True)
    assert_same(a, b)
    assert a["step"] == parent["step"] + plan.steps
    assert a["contract"]["auxiliary_reconstruction"]["weight"] == .2
    assert not a["contract"]["ema"] and a["contract"]["first_frame_focus"]["history_noise"] == 0
    rows = [json.loads(line) for line in (paused / "history.jsonl").read_text().splitlines()]
    assert rows[0]["lr"] == pytest.approx(2e-5 / 50)
    assert all(row["epoch"] == 4 for row in rows)
    assert "t2v.first" in rows[0]["loss"] and "t2v.future" in rows[0]["loss"]
    seen = Counter(sample["sample_id"] for row in rows for sample in row["rank0_samples"]
                   if not sample["padding"] and sample["task"] != "image_reconstruction")
    assert len(seen) == 8 and set(seen.values()) == {1}
    old = dict(paused_checkpoint)
    old.pop("rng_states")
    with pytest.raises(ValueError, match="RNG"):
        trainer.validate_resume(old, a["contract"], a["connector_spec"])
    assert (full / f"stage2_dit_connector_step{parent['step'] + 4:06d}.pt").is_file()
    assert "optimizer" not in torch.load(full / "stage2_dit_connector_latest.pt", weights_only=True)


def test_budget_tool_and_launcher_safety(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage1_summary.json").write_text("{}")
    plan = media.MediaEpochPlan(3, 5, 2, 2, epochs=1, start_epoch=3)
    parent = tmp_path / "parent.pt"
    torch.save(parent_metadata(plan, sha256_file(data / "stage1_summary.json")), parent)
    monkeypatch.setattr(planner, "plan_from_release", lambda *a, **kw: plan)
    args = SimpleNamespace(parent=parent, parent_step=3 * plan.steps_per_epoch, parent_sha256=sha256_file(parent),
                           data_root=data, world_size=2, accumulation=2, seed=plan.seed, min_train_videos=1,
                           require_expanded_release=False, integration_smoke=False, output=tmp_path / "out/plan.json",
                           audit_samples=0)
    report = planner.plan(args)
    assert report["additional_updates"] == plan.steps and report["absolute_final_step"] == 4 * plan.steps_per_epoch
    assert report["loss_weights"] == dict(t2i=.5, video=.5, reconstruction=.2)
    assert not report["caption_semantics_verified"] and not report["teacher"]
    args.parent_sha256 = "a" * 64
    with pytest.raises(ValueError, match="SHA256"):
        planner.plan(args)

    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    out = tmp_path / "job"
    out.mkdir()
    env = dict(os.environ, PROJECT_ROOT=str(root), SLURM_JOB_ID="12345", OUT=str(out),
               INIT_STAGE2=str(parent), DATA_ROOT=str(data), CONDA_ENV=str(tmp_path),
               PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls), TMPDIR=str(tmp_path / "tmp"))
    for key in ("STEPS", "RESUME", "INIT_CONNECTOR", "AUTO_CONTINUE"):
        env.pop(key, None)
    for name, body in {
        "module": "exit 0",
        "python": 'printf "PREFLIGHT %s\\n" "$*" >> "$CALLS"; exit "${PREFLIGHT_RC:-0}"',
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"; exit "${TRAIN_RC:-0}"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s OUT=%s\\n" "$*" "$RESUME" "$OUT" >> "$CALLS"; echo 12346',
    }.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    script = root / "scripts/train_mobileov_stage2_first_frame_focus_1node8gpu.sbatch"
    def launch(**updates):
        return subprocess.run(["bash", str(script)], env={**env, **updates}, capture_output=True, text=True)
    assert launch().returncode == 0
    text = calls.read_text()
    assert "--media-epochs 1 --media-start-epoch 3 --t2v-first-probability .5" in text
    assert "--reconstruction-weight .2" in text and "--init-connector" not in text
    assert "CONTINUE" not in text
    (out / "training_paused.json").write_text("{}")
    (out / "stage2_resume.pt").write_text("checkpoint")
    assert launch().returncode == 0
    assert "--dependency=afterok:12345 --export=ALL" in calls.read_text()
    calls.unlink()
    assert launch(TRAIN_RC="1").returncode != 0 and "CONTINUE" not in calls.read_text()
    assert launch(PREFLIGHT_RC="1").returncode != 0
    assert launch(STEPS="100000").returncode != 0
    assert launch(INIT_STAGE2=str(tmp_path / "missing.pt")).returncode != 0
    for name in (script.name, "smoke_mobileov_stage2_first_frame_focus_local.sbatch"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
