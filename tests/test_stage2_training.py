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
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training import stage2_alignment as stage2
from tools.train_mobileov_stage1 import parse_args, save, runtime_expired


def scheduler():
    return SimpleNamespace(config=SimpleNamespace(stages=3, num_train_timesteps=10),
                           sigmas_per_stage={s: torch.linspace(1, .1, 10) for s in range(3)},
                           timesteps_per_stage={s: torch.linspace(1000, 100, 10) for s in range(3)},
                           start_sigmas={0: 1., 1: .8, 2: .5}, end_sigmas={0: .67, 1: .33, 2: 0.})


def install_history():
    sys.modules["neodragon.utils.generation_utils"] = SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, sample, encoder_hidden_states, pooled_projections, **kwargs):
        signal = encoder_hidden_states[..., 0].mean() + pooled_projections.mean()
        return [sample[0][-1] * self.weight + signal]


def batch(rank, step, micro):
    rng = torch.Generator().manual_seed(900 + rank * 9 + step * 2 + micro)
    length = 9 + rank + micro
    return [torch.randn(1, length, 8, generator=rng) for _ in range(3)], torch.ones(1, length), torch.randn(
        1, 4, 7 if micro else 1, 16, 24, generator=rng)


def test_phase_defaults_and_invalid_initialization():
    phase1 = parse_args(["--output-dir", "unused", "--steps", "100000"])
    assert phase1.phase == 1 and phase1.accumulation == 3 and phase1.lr == 1e-4
    assert phase1.vae_window_size == 16
    phase2 = parse_args(["--phase", "2", "--init-connector", "init.pt", "--output-dir", "unused"])
    assert phase2.steps == 100000 and phase2.accumulation == 2 and phase2.lr == 2e-5
    assert phase2.save_every == phase2.archive_every == 10000
    assert phase2.tasks == ("t2i", "t2v")
    assert phase2.vae_window_size == 8
    for extra in (["--phase", "2"], ["--init-connector", "init.pt"], ["--max-runtime-seconds", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["--output-dir", "unused", *extra])


def test_init_verifies_backbones_and_keeps_fp32_masters():
    connector, dit = Stage1Connector(8, 16, 12), TinyDiT().bfloat16().requires_grad_(False)
    payload = dict(format="mobileov_stage1_mcp_v1", step=15000, connector_spec=connector.spec,
                   connector=copy.deepcopy(connector.state_dict()),
                   contract=dict(config_sha256="cfg", processor_sha256="proc",
                                 frozen_weights_sha256={"dit": "native"}, short_side=320,
                                 long_side=512, max_tokens=2048))
    args = SimpleNamespace(short_side=320, long_side=512, max_tokens=2048, gradient_checkpointing=True)
    kwargs = dict(expected_step=15000, signatures={"dit": "native"}, config_sha256="cfg",
                  processor_sha256="proc", args=args)
    stage2.initialize_from_alignment(payload, connector, dit, **kwargs)
    assert all(p.requires_grad and p.dtype == torch.float32 for m in (connector, dit) for p in m.parameters())
    assert dit.gradient_checkpointing and dit.gradient_checkpointing_ratio == 0
    with pytest.raises(ValueError, match="mismatch"):
        stage2.initialize_from_alignment(payload, connector, dit, **{**kwargs, "signatures": {"dit": "other"}})
    with pytest.raises(ValueError, match="step"):
        stage2.initialize_from_alignment(payload, connector, dit, **{**kwargs, "expected_step": 100000})
    payload["connector"]["pooled_head.1.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        stage2.initialize_from_alignment(payload, connector, dit, **kwargs)


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("unit", [0, 1, 6])
def test_both_components_update_at_all_stages_and_units(stage, unit, monkeypatch):
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    torch.manual_seed(42)
    model = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss = model(*batch(0, 0, 1), scheduler(), stage=stage, unit=unit,
                 generator=torch.Generator().manual_seed(2))
    loss.backward()
    assert all(bool(stage2.gradient_norm(m) > 0) for m in (model.connector, model.dit))
    optimizer.step()
    for component in ("connector", "dit"):
        assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()
                   if key.startswith(component + "."))


def test_checkpoint_contents_hardlink_and_exact_optimizer_resume(tmp_path):
    connector, dit = Stage1Connector(8, 16, 12), TinyDiT()
    model = stage2.JointFlowModel(connector, dit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    args = SimpleNamespace(output_dir=tmp_path, archive_every=10000)
    save(args, 10000, connector, optimizer, dict(format=stage2.FORMAT), dit=dit)
    latest = tmp_path / "stage2_dit_connector_latest.pt"
    archive = tmp_path / "stage2_dit_connector_step010000.pt"
    assert latest.stat().st_ino == archive.stat().st_ino
    inference = torch.load(latest, weights_only=True)
    assert set(inference) == {"format", "step", "connector_spec", "connector", "dit", "contract"}
    resume = torch.load(tmp_path / "stage2_resume.pt", weights_only=True)
    restored = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
    restored.connector.load_state_dict(resume["connector"])
    restored.dit.load_state_dict(resume["dit"])
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.1)
    opt2.load_state_dict(resume["optimizer"])
    for current, opt in ((model, optimizer), (restored, opt2)):
        opt.zero_grad()
        sum(p.square().sum() for p in current.parameters()).backward()
        opt.step()
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    save(args, 10001, connector, optimizer, dict(format=stage2.FORMAT), dit=dit)
    assert torch.load(archive, weights_only=True)["step"] == 10000
    assert torch.load(latest, weights_only=True)["step"] == 10001


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    install_history()
    dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        module = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
        model = torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True,
                                                         gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with model.no_sync() if micro == 0 else nullcontext():
                    loss = model(*batch(rank, step, micro), scheduler(), stage=(step + micro + rank) % 3,
                                 unit=6 if micro else 0, generator=torch.Generator().manual_seed(700 + micro + rank))
                    (loss / 2).backward()
            optimizer.step()
        torch.save(module.state_dict(), Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_joint_ddp_accumulation_matches_global_batch(tmp_path, monkeypatch):
    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    torch.manual_seed(42)
    reference = stage2.JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
    optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-5)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(2):
            for micro in range(2):
                loss = reference(*batch(rank, step, micro), scheduler(), stage=(step + micro + rank) % 3,
                                 unit=6 if micro else 0, generator=torch.Generator().manual_seed(700 + micro + rank))
                (loss / 4).backward()
        optimizer.step()
    a = torch.load(tmp_path / "rank0.pt", weights_only=True)
    b = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, value in reference.state_dict().items():
        assert torch.equal(a[key], b[key]), key
        assert torch.allclose(a[key], value, atol=1e-6, rtol=1e-4), key


def test_runtime_budget_and_new_scripts(monkeypatch):
    monkeypatch.setattr("tools.train_mobileov_stage1.time.monotonic", lambda: 100.)
    ctx = SimpleNamespace(device="cpu", is_distributed=False)
    assert runtime_expired(SimpleNamespace(max_runtime_seconds=50), ctx, 0)
    assert not runtime_expired(SimpleNamespace(max_runtime_seconds=0), ctx, 0)
    root = Path(__file__).resolve().parents[1]
    for name in ("train_mobileov_stage1_alignment_100k_1node8gpu.sbatch",
                 "train_mobileov_stage2_from_alignment15k_1node8gpu.sbatch", "smoke_mobileov_stage2_local.sbatch"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)


@pytest.mark.parametrize("phase", [1, 2])
def test_launchers_pass_correct_recipe_and_only_continue_after_pause(tmp_path, phase):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    env = dict(os.environ, PROJECT_ROOT=str(root), CONDA_ENV=str(tmp_path),
               SLURM_JOB_ID="12345", OUT=str(tmp_path / "out"), DATA_ROOT=str(tmp_path / "data"),
               HF_HOME=str(tmp_path / "hf"), TMPDIR="/tmp", PATH=f"{bin_dir}:{os.environ['PATH']}",
               CALLS=str(calls))
    for key in ("RESUME", "STEPS", "LR", "SAVE_EVERY", "ARCHIVE_EVERY", "ACCUMULATION", "INIT_CONNECTOR"):
        env.pop(key, None)
    Path(env["DATA_ROOT"]).mkdir()
    (Path(env["DATA_ROOT"]) / ".stage1_data_complete").write_text("ready")
    Path(env["OUT"]).mkdir()
    init = tmp_path / "init.pt"
    init.write_text("present")
    if phase == 2:
        env["INIT_CONNECTOR"] = str(init)
    for name, body in {
        "module": "exit 0",
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s\\n" "$*" "$RESUME" >> "$CALLS"',
    }.items():
        script = bin_dir / name
        script.write_text("#!/bin/bash\n" + body + "\n")
        script.chmod(0o755)
    script = root / "scripts" / ("train_mobileov_stage1_alignment_100k_1node8gpu.sbatch" if phase == 1 else
                                  "train_mobileov_stage2_from_alignment15k_1node8gpu.sbatch")
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    text = calls.read_text()
    assert f"--phase {phase}" in text and "--save-every 10000 --archive-every 10000" in text
    assert "--steps 100000" in text
    assert "CONTINUE" not in text
    assert ("--init-connector" in text) == (phase == 2)
    out = Path(env["OUT"])
    (out / "training_paused.json").write_text("{}")
    (out / f"stage{phase}_resume.pt").write_text("checkpoint")
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert "CONTINUE --dependency=afterok:12345 --export=ALL" in calls.read_text()
    (out / "training_complete.json").write_text("{}")
    calls.unlink()
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert "CONTINUE" not in calls.read_text()
    if phase == 2:
        init.unlink()
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True)
        assert result.returncode != 0


@pytest.mark.parametrize("stage,unit", [(0, 0), (1, 0), (2, 0), (0, 6), (1, 6), (2, 6)])
def test_actual_pyramid_block_checkpointing_on_cpu(stage, unit, monkeypatch):
    repo = Path(__file__).resolve().parents[1] / "checkpoints" / "neodragon_repo"
    if not repo.exists():
        pytest.skip("Optional native NeoDragon source checkout is not installed")
    monkeypatch.syspath_prepend(str(repo))
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    torch.manual_seed(19)
    dit = PyramidMMDiT(sample_size=16, in_channels=4, num_layers=2, num_attention_heads=2,
                       attention_head_dim=16, caption_projection_dim=32, pooled_projection_dim=12,
                       pos_embed_max_size=32, use_gradient_checkpointing=True, gradient_checkpointing_ratio=0.)
    # Released weights are nonzero; randomize zero-initialized output/modulation layers for this small fixture.
    for module in dit.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.03)
    connector = Stage1Connector(8, 32, 12)
    model = stage2.JointFlowModel(connector, dit).train()
    sched = PyramidFlowMatchEulerDiscreteScheduler()
    layers, mask, latent = batch(0, 0, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model([x.bfloat16() for x in layers], mask, latent.bfloat16(), sched, stage=stage, unit=unit,
                     generator=torch.Generator().manual_seed(2))
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(bool(stage2.gradient_norm(m) > 0) for m in (dit, connector))


def test_training_loop_pause_resume_is_exact_for_both_phases(tmp_path, monkeypatch):
    """Exercise the actual training loop with small CPU modules, not a GPU smoke replacement."""
    from tools import train_mobileov_stage1 as trainer
    from new_mobile_ov.training.distributed import DistributedContext
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    root = tmp_path / "data"
    root.mkdir()
    (root / "stage1_summary.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text("test")
    smol_file = tmp_path / "smol.pt"
    smol_file.write_text("test")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(smol_file)))

    def load_stack(args, device):
        encoder, vae, dit = nn.Linear(2, 2), nn.Linear(2, 2), TinyDiT()
        for module in (encoder, vae, dit):
            module.eval().requires_grad_(False)
        encoder.processor_sha256 = "test_processor"
        return cfg, encoder, Stage1Connector(8, 16, 12), dit, vae, scheduler()

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, *, steps, accumulation, start_step, tasks, **kwargs):
            self.steps, self.accumulation, self.start, self.tasks = steps, accumulation, start_step, tasks
            self.records = {task: list(range(10)) for task in tasks}

        def __len__(self):
            return (self.steps - self.start) * self.accumulation

        def __getitem__(self, index):
            micro = self.start * self.accumulation + index
            return dict(micro=micro, task=self.tasks[micro % len(self.tasks)], sample_id=str(micro))

    def encode(encoder, vae, sample, device, rng, dropout, *, window_size):
        layers = [torch.randn(1, 9, 8, generator=rng) for _ in range(3)]
        latent = torch.randn(1, 4, 7 if sample["task"] == "t2v" else 1, 16, 24, generator=rng)
        return layers, torch.ones(1, 9), latent, False

    monkeypatch.setattr(trainer, "load_stack", load_stack)
    monkeypatch.setattr(trainer, "Stage1Dataset", Dataset)
    monkeypatch.setattr(trainer, "verify_release", lambda root: None)
    monkeypatch.setattr(trainer, "validate", lambda *args: {})
    monkeypatch.setattr(trainer, "encode_sample", encode)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    common = ["--config", str(config), "--data-root", str(root), "--workers", "0", "--verify-frozen"]

    def train(name, extra):
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        trainer.train(parse_args([*common, "--output-dir", str(out), *extra]), ctx)
        return out

    alignment = train("align", ["--steps", "100000", "--max-runtime-seconds", "0.0001"])
    assert json.loads((alignment / "training_paused.json").read_text())["step"] == 1
    train("align", ["--steps", "100000", "--stop-after", "3", "--resume", str(alignment / "stage1_resume.pt")])
    assert json.loads((alignment / "training_paused.json").read_text())["step"] == 3
    assert not (alignment / "training_complete.json").exists()
    joint_args = ["--phase", "2", "--steps", "4", "--init-connector", str(alignment / "stage1_connector_latest.pt"),
                  "--expected-init-step", "3"]
    partial = train("joint", [*joint_args, "--stop-after", "2"])
    train("joint", [*joint_args, "--resume", str(partial / "stage2_resume.pt")])
    full = train("joint_full", joint_args)
    assert (partial / "training_complete.json").exists() and not (partial / "training_paused.json").exists()
    a = torch.load(partial / "stage2_dit_connector_latest.pt", weights_only=True)
    b = torch.load(full / "stage2_dit_connector_latest.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    for component in ("connector", "dit"):
        assert all(torch.equal(v, b[component][k]) for k, v in a[component].items())
    rows = [json.loads(line) for line in (partial / "history.jsonl").read_text().splitlines()]
    assert all(set(row["loss"]) == {"t2i", "t2v"} for row in rows)

    # A completed run can grow its budget without resetting Adam, warmup or sample order.
    resume_args = [*joint_args, "--steps", "6", "--resume", str(partial / "stage2_resume.pt"),
                   "--expected-resume-step", "4"]
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        train("extension_rejected", resume_args)
    extended = train("extended", [*resume_args, "--extend-steps"])
    reference = train("reference6", [*joint_args, "--steps", "6"])
    left = torch.load(extended / "stage2_resume.pt", weights_only=True)
    right = torch.load(reference / "stage2_resume.pt", weights_only=True)

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        else:
            assert a == b

    same(left, right)
    assert torch.load(partial / "stage2_resume.pt", weights_only=True)["step"] == 4
    provenance = json.loads((extended / "resume_from_step000004.json").read_text())
    assert provenance["changed_contract_fields"] == ["steps"] and provenance["optimizer_restored"]
    assert provenance["previous_target_steps"] == 4 and provenance["target_steps"] == 6
    extended_rows = [json.loads(line) for line in (extended / "history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in extended_rows] == [5, 6]
    assert extended_rows[0]["lr"] == pytest.approx(2e-5 * 5 / 50)


def test_entrypoint_records_root_exception_without_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    error_file = tmp_path / "worker_error.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCHELASTIC_ERROR_FILE=str(error_file))
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run([sys.executable, "-s", str(root / "tools/train_mobileov_stage1.py"),
                             "--output-dir", str(tmp_path / "unused")], cwd=root, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    error = json.loads(error_file.read_text())
    assert "GPU training must run inside srun/sbatch" in error["message"]["message"]
    assert "RuntimeError" in error["message"]["extraInfo"]["py_callstack"]
