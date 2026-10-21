import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training import stage2_epoch_extension as extension
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training.distributed import DistributedContext
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from tools import train_mobileov_stage1 as trainer
from tools import plan_mobileov_stage2_epoch_extension as planner
from tools.data_prepare.download_alignment_images import sha256_file
from test_stage2_training import TinyDiT, scheduler


def cli(resume="parent.pt", out="unused", epochs=5):
    return ["--phase", "2", "--resume", str(resume), "--output-dir", str(out),
            "--media-epochs", str(epochs), "--extend-media-epochs", "--reconstruction-weight", ".1"]


def test_cli_requires_explicit_optimizer_resume_and_unchanged_recipe():
    args = trainer.parse_args(cli())
    assert args.lr == 2e-5 and args.warmup == 50 and args.accumulation == 2
    assert not args.init_stage2 and not args.init_connector
    for extra in (["--init-stage2", "weights.pt"], ["--init-connector", "align.pt"],
                  ["--reconstruction-weight", ".2"], ["--t2v-first-probability", ".5"],
                  ["--history-corrupt-max", ".1"], ["--media-start-epoch", "3"],
                  ["--extend-steps"], ["--steps", "473295"], ["--media-epochs", "3"]):
        with pytest.raises(SystemExit):
            trainer.parse_args([*cli(), *extra])
    for values in (["--phase", "2", "--output-dir", "x", "--extend-media-epochs", "--media-epochs", "5"],
                   ["--phase", "2", "--init-connector", "a.pt", "--output-dir", "x", "--media-epochs", "5"]):
        with pytest.raises(SystemExit):
            trainer.parse_args(values)


@pytest.mark.parametrize("epochs", [4, 5])
def test_total_budget_and_same_sampler_prefix_as_continuous_training(epochs):
    small = media.MediaEpochPlan(7, 11, 2, 2, epochs=epochs, allow_partial_cycle=True)
    full = media.MediaEpochPlan(7, 11, 2, 2, epochs=6)
    for step in range(small.steps):
        for rank in range(2):
            for micro in range(2):
                assert small.slot(step, micro, rank) == full.slot(step, micro, rank)
    plan = media.MediaEpochPlan(984636, 529902, 8, 2, epochs=epochs, allow_partial_cycle=True)
    assert plan.steps == {4: 378636, 5: 473295}[epochs]
    assert "per_video_mode_presentations" not in plan.contract()
    assert sum(plan.contract()["presentations_by_video_mode"].values()) == 529902 * epochs


@pytest.fixture
def training_environment(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage1_summary.json").write_text("{}")
    config, smol = tmp_path / "config", tmp_path / "smol"
    config.write_text("config")
    smol.write_text("smol")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(smol)))
    stochastic = {"enabled": False}

    class StochasticDiT(TinyDiT):
        def forward(self, *args, **kwargs):
            value = super().forward(*args, **kwargs)[0]
            if stochastic["enabled"]:
                value = torch.nn.functional.dropout(value, .2, training=self.training)
            return [value]

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
        rng = torch.Generator().manual_seed(700 + sample["micro"])
        return [torch.randn(1, 13, 8, generator=rng) for _ in range(3)], torch.ones(1, 13)

    def plan(*args, epochs=3, **kwargs):
        return media.MediaEpochPlan(17, 19, 1, 2, epochs=epochs, allow_partial_cycle=epochs in (4, 5))

    monkeypatch.setattr(trainer, "load_stack", stack)
    monkeypatch.setattr(trainer, "verify_release", lambda root: None)
    monkeypatch.setattr(media, "plan_from_release", plan)
    monkeypatch.setattr(media, "MediaEpochDataset", Dataset)
    monkeypatch.setattr(trainer, "encode_sample", encode)
    monkeypatch.setattr(trainer, "validate", lambda *args: {})
    monkeypatch.setattr(trainer.stage2, "reconstruction_condition", reconstruction)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    torch.manual_seed(20260911)
    _, encoder, connector, dit, vae, _ = stack(None, None)
    init = tmp_path / "init.pt"
    torch.save(dict(format=trainer.FORMAT, step=15000, connector_spec=connector.spec,
                    connector=connector.state_dict(), contract=dict(config_sha256=sha256_file(config),
                    processor_sha256="processor", data_summary_sha256="alignment_data",
                    frozen_weights_sha256=frozen_signatures(dict(smolvlm2=encoder, dit=dit, vae=vae)),
                    short_side=320, long_side=512, max_tokens=2048)), init)
    common = ["--config", str(config), "--data-root", str(data), "--workers", "0", "--log-every", "1",
              "--save-every", "1000", "--archive-every", "1000", "--validate-every", "1000", "--verify-frozen"]
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))

    def run(name, *, resume=None, epochs=3, extra=()):
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        values = (cli(resume, out, epochs) if resume else
                  ["--phase", "2", "--init-connector", str(init), "--output-dir", str(out),
                   "--media-epochs", "3", "--reconstruction-weight", ".1"])
        trainer.train(trainer.parse_args([*values, *common, *extra]), ctx)
        return out

    return SimpleNamespace(root=tmp_path, run=run, plan=plan, data=data, stochastic=stochastic)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def test_original_three_epochs_extend_to_five_equal_uninterrupted_and_pause_resume(training_environment, monkeypatch):
    env = training_environment
    parent = env.run("parent")
    source = parent / "stage2_resume.pt"
    original_bytes = source.read_bytes()
    original_contract = (parent / "run_contract.json").read_bytes()
    full_plan = env.plan(epochs=5)
    parent_step, fourth_step = 3 * full_plan.steps_per_epoch, 4 * full_plan.steps_per_epoch
    with monkeypatch.context() as patch:
        patch.setattr(media, "plan_from_release", lambda *args, **kwargs: full_plan)
        reference = env.run("continuous")
    extended = env.run("extended", resume=source, epochs=5)
    interrupted = env.run("interrupted", resume=source, epochs=5, extra=["--stop-after", str(parent_step + 2)])
    env.run("interrupted", resume=interrupted / "stage2_resume.pt", epochs=5)
    read = lambda root: torch.load(root / "stage2_resume.pt", weights_only=True)
    expected, actual, resumed = read(reference), read(extended), read(interrupted)
    for key in ("connector", "dit", "optimizer"):
        same(expected[key], actual[key])
        same(actual[key], resumed[key])
    assert actual["step"] == resumed["step"] == full_plan.steps
    assert source.read_bytes() == original_bytes
    assert (parent / "run_contract.json").read_bytes() == original_contract
    assert (extended / f"stage2_dit_connector_step{fourth_step:06d}.pt").is_file()
    assert (extended / f"stage2_dit_connector_step{full_plan.steps:06d}.pt").is_file()
    assert json.loads((extended / "epoch4_complete.json").read_text())["step"] == fourth_step
    receipt = json.loads((extended / f"resume_from_step{parent_step:06d}.json").read_text())
    assert receipt["changed_contract_fields"] == ["media_epochs", "steps"]
    assert receipt["epoch_extension"]["max_adam_step"] == parent_step
    assert receipt["epoch_extension"]["legacy_global_rng_unavailable"]
    assert len(actual["rng_states"]) == 1
    audit = json.loads((extended / f"first_update_optimizer_audit_step{parent_step + 1:06d}.json").read_text())
    assert audit["max_adam_step"] == parent_step + 1 and audit["lr"] == 2e-5 and not audit["warmup_restarted"]
    rows = [json.loads(line) for line in (extended / "history.jsonl").read_text().splitlines()]
    assert {row["lr"] for row in rows} == {2e-5}
    assert rows[0]["rank0_samples"][0]["epoch"] == 4


def test_can_stop_at_four_then_extend_to_five(training_environment):
    env = training_environment
    parent = env.run("parent")
    fourth = env.run("fourth", resume=parent / "stage2_resume.pt", epochs=4)
    fifth = env.run("fifth", resume=fourth / "stage2_resume.pt", epochs=5)
    direct = env.run("direct", resume=parent / "stage2_resume.pt", epochs=5)
    a, b = [torch.load(root / "stage2_resume.pt", weights_only=True) for root in (fifth, direct)]
    for key in ("connector", "dit", "optimizer"):
        same(a[key], b[key])


def test_new_extension_restores_rng_for_stochastic_layers(training_environment, monkeypatch):
    env = training_environment
    env.stochastic["enabled"] = True
    def validation(*args):
        torch.rand(13)
        return {}
    monkeypatch.setattr(trainer, "validate", validation)
    parent = env.run("parent")
    direct = env.run("direct", resume=parent / "stage2_resume.pt", epochs=5)
    paused = env.run("paused", resume=parent / "stage2_resume.pt", epochs=5,
                     extra=["--stop-after", str(env.plan().steps + 2)])
    env.run("paused", resume=paused / "stage2_resume.pt", epochs=5)
    a, b = [torch.load(root / "stage2_resume.pt", weights_only=True) for root in (direct, paused)]
    for key in ("connector", "dit", "optimizer", "rng_states"):
        same(a[key], b[key])


def test_reject_inference_only_wrong_recipe_and_reset_optimizer(training_environment):
    env = training_environment
    parent = env.run("parent")
    with pytest.raises(ValueError, match="Missing original Adam"):
        env.run("weights_only", resume=parent / "stage2_dit_connector_latest.pt", epochs=5)
    payload = torch.load(parent / "stage2_resume.pt", weights_only=True)
    for case in ("reset", "lr", "loss", "partial", "history", "nan", "shape"):
        bad = copy.deepcopy(payload)
        if case == "reset":
            for value in bad["optimizer"]["state"].values():
                value["step"] = torch.tensor(1.)
        elif case == "lr":
            bad["optimizer"]["param_groups"][0]["lr"] = 1e-4
        elif case == "loss":
            bad["contract"]["auxiliary_reconstruction"]["weight"] = .2
        elif case == "partial":
            bad["step"] = 12
        elif case == "history":
            bad["contract"]["history_corruption"] = {}
        elif case == "nan":
            next(iter(bad["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
        else:
            value = next(iter(bad["optimizer"]["state"].values()))
            value["exp_avg"] = value["exp_avg_sq"] = torch.zeros(123)
        path = env.root / f"{case}.pt"
        torch.save(bad, path)
        with pytest.raises(ValueError):
            env.run(f"reject_{case}", resume=path, epochs=5)
    with pytest.raises(ValueError, match="Preserve the original|never rewind"):
        env.run("parent", resume=parent / "stage2_resume.pt", epochs=5)
    existing = env.run("existing", resume=parent / "stage2_resume.pt", epochs=5,
                       extra=["--stop-after", str(env.plan().steps + 1)])
    with pytest.raises(ValueError, match="never rewind"):
        env.run("existing", resume=parent / "stage2_resume.pt", epochs=5)


def test_preflight_checks_actual_adam_state_and_protects_output(training_environment, monkeypatch):
    env = training_environment
    parent = env.run("parent")
    monkeypatch.setattr(planner, "plan_from_release", env.plan)
    args = SimpleNamespace(resume=parent / "stage2_resume.pt", data_root=env.data,
                           output_dir=env.root / "preflight", epochs=5, world_size=1, accumulation=2)
    report = planner.run(args)
    assert report["resume_step"] == env.plan().steps
    assert report["target_step"] == env.plan(epochs=5).steps
    assert report["max_adam_step"] == report["resume_step"] and not report["warmup_restarted"]
    assert not report["global_rng_available"]
    planner.run(args)
    with pytest.raises(ValueError, match="Use a new output"):
        planner.run(SimpleNamespace(**dict(vars(args), output_dir=parent)))
    original = json.loads((parent / "run_contract.json").read_text())
    target = dict(original, steps=env.plan(epochs=5).steps, media_epochs=env.plan(epochs=5).contract())
    (args.output_dir / "run_contract.json").write_text(json.dumps(target))
    with pytest.raises(ValueError, match="never rewind"):
        planner.run(args)
    (args.output_dir / "run_contract.json").write_text("{}")
    with pytest.raises(ValueError, match="different run"):
        planner.run(args)
    with pytest.raises(FileNotFoundError):
        planner.run(SimpleNamespace(**dict(vars(args), resume=env.root / "missing.pt")))


def test_gpu_launcher_checks_state_and_continues_own_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    source = tmp_path / "stage2_resume.pt"
    source.write_text("optimizer_checkpoint")
    out = tmp_path / "out"
    out.mkdir()
    for name, body in {
        "module": "exit 0",
        "python": 'printf "PREFLIGHT %s\\n" "$*" >> "$CALLS"; exit "${PREFLIGHT_RC:-0}"',
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"; exit "${TRAIN_RC:-0}"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s OUT=%s TOTAL_EPOCHS=%s\\n" "$*" "$RESUME" "$OUT" "$TOTAL_EPOCHS" >> "$CALLS"; echo 999',
    }.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", PROJECT_ROOT=str(root),
               SLURM_JOB_ID="123", CONDA_ENV=str(tmp_path), RESUME=str(source), OUT=str(out),
               TMPDIR=str(tmp_path / "tmp"), CALLS=str(calls))
    for key in ("STEPS", "INIT_CONNECTOR", "INIT_STAGE2", "LR", "TOTAL_EPOCHS", "AUTO_CONTINUE"):
        env.pop(key, None)
    script = root / "scripts/train_mobileov_stage2_extend_epochs_1node8gpu.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    def run(**updates):
        return subprocess.run(["bash", str(script)], env={**env, **updates}, capture_output=True, text=True)
    assert run().returncode == 0
    assert "--media-epochs 5 --extend-media-epochs" in calls.read_text()
    assert "--init-stage2" not in calls.read_text() and "--init-connector" not in calls.read_text()
    (out / "training_paused.json").write_text("{}")
    (out / "stage2_resume.pt").write_text("own_optimizer")
    assert run(TOTAL_EPOCHS="4").returncode == 0
    assert f"RESUME={out}/stage2_resume.pt OUT={out} TOTAL_EPOCHS=4" in calls.read_text()
    calls.unlink()
    assert run(PREFLIGHT_RC="1").returncode != 0
    assert "TRAIN" not in calls.read_text() and "CONTINUE" not in calls.read_text()
    calls.unlink()
    assert run(TRAIN_RC="1").returncode != 0
    assert "CONTINUE" not in calls.read_text()
    assert run(RESUME=str(tmp_path / "missing.pt")).returncode != 0
    assert run(LR="1e-4").returncode != 0
    assert run(TOTAL_EPOCHS="6").returncode != 0
