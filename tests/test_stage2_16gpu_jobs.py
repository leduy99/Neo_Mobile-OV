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
from new_mobile_ov.training import stage2_epoch_extension as extension
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training.distributed import DistributedContext
from tools import train_mobileov_stage1 as trainer
from tools import plan_mobileov_stage2_epoch_extension as planner
from test_stage2_epoch_extension_resume import training_environment, same
from test_stage2_training import TinyDiT, scheduler

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mobileov_stage2_2node16gpu.sh"
RESUME_JOB = ROOT / "scripts/train_mobileov_stage2_resume_epoch3_to5_2node16gpu.sbatch"
HISTORY_JOB = ROOT / "scripts/train_mobileov_stage2_history_fresh5epochs_2node16gpu.sbatch"


def test_16gpu_keeps_media_noise_and_objective_weights_of_8gpu():
    old = media.MediaEpochPlan(984636, 529902, 8, 2, epochs=6)
    new = media.MediaEpochPlan(984636, 529902, 16, 1, epochs=5,
                               allow_partial_cycle=True, rng_world_size=8)
    assert new.steps_per_epoch == old.steps_per_epoch == 94659
    assert new.steps == 473295
    for step in (0, 94658, 94659, 283976, 283977, 378635, 473294):
        for rank in range(16):
            a, b = old.slot(step, rank // 8, rank % 8), new.slot(step, 0, rank)
            assert b.pop("rng_rank") == rank % 8
            assert a == b
            for seed in (20260911, 20260911 + 444444, 20260911 + 777777):
                before = trainer.generator_for("cpu", seed, a["micro"], rank % 8)
                after = trainer.generator_for("cpu", seed, b["micro"], rank % 8)
                assert torch.equal(torch.randn(11, generator=before), torch.randn(11, generator=after))
            # DDP averaging and gradient accumulation keep the coefficient per example.
            assert a["loss_scale"] / 8 / 2 == b["loss_scale"] / 16
    with pytest.raises(ValueError, match="RNG world size"):
        media.MediaEpochPlan(2, 3, 16, 1, rng_world_size=3)


def fresh_cli(init="init.pt", out="unused", epochs=5):
    return ["--phase", "2", "--init-connector", str(init), "--output-dir", str(out),
            "--media-epochs", str(epochs), "--fresh-media-history", "--history-corrupt-max", str(1/3),
            "--reconstruction-weight", ".1"]


def test_fresh_history_cli_does_not_enable_old_fork_changes():
    args = trainer.parse_args([*fresh_cli(), "--accumulation", "1", "--media-rng-world-size", "8"])
    assert args.lr == 2e-5 and args.warmup == 50 and args.history_corrupt_max == 1/3
    assert args.t2v_first_probability is None and not args.init_stage2 and not args.extend_media_epochs
    assert args.reconstruction_weight == .1 and args.media_start_epoch == 0
    for extra in (["--history-corrupt-max", "0"], ["--reconstruction-weight", ".2"],
                  ["--media-epochs", "4"], ["--t2v-first-probability", ".5"],
                  ["--extend-media-epochs"], ["--repartition-media-batch"]):
        with pytest.raises(SystemExit):
            trainer.parse_args([*fresh_cli(), *extra])


def test_repartition_preflight_preserves_adam_and_requires_same_batch(training_environment, monkeypatch):
    env = training_environment
    parent = env.run("parent")
    saved = torch.load(parent / "stage2_resume.pt", weights_only=True)
    plan = media.MediaEpochPlan(17, 19, 2, 1, epochs=5, allow_partial_cycle=True, rng_world_size=1)
    target = dict(saved["contract"], world_size=2, accumulation=1, steps=plan.steps, media_epochs=plan.contract())
    extension.validate_extension(saved["contract"], target, saved["step"], repartition=True)
    with pytest.raises(ValueError, match="repartition"):
        extension.validate_extension(saved["contract"], target, saved["step"])
    wrong = media.MediaEpochPlan(17, 19, 2, 2, epochs=5, allow_partial_cycle=True, rng_world_size=1)
    with pytest.raises(ValueError, match="same global batch"):
        extension.validate_extension(saved["contract"], dict(target, accumulation=2,
                                     steps=wrong.steps, media_epochs=wrong.contract()), saved["step"], repartition=True)
    monkeypatch.setattr(planner, "plan_from_release", lambda *a, **kw: plan)
    args = SimpleNamespace(resume=parent / "stage2_resume.pt", data_root=env.data, epochs=5,
                           output_dir=env.root / "preflight16", world_size=2, accumulation=1,
                           media_rng_world_size=1, repartition_media_batch=True)
    report = planner.run(args)
    assert report["ddp_repartitioned"] and report["source_world_size"] == 1 and report["world_size"] == 2
    assert report["max_adam_step"] == saved["step"] and report["lr"] == 2e-5


def migrated_trainer_worker(rank, directory, mode="resume"):
    """Scale 1 rank x 2 micros to 2 ranks x 1 micro through the real trainer."""
    torch.set_num_threads(1)
    directory = Path(directory)
    sys.modules["neodragon.utils.generation_utils"] = SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        def stack(args, device):
            encoder, vae, dit = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), TinyDiT().bfloat16()
            for module in (encoder, vae, dit):
                module.eval().requires_grad_(False)
            encoder.processor_sha256 = "processor"
            cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(directory / "smol")))
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
            return ([torch.randn(1, 9, 8, generator=rng) for _ in range(3)], torch.ones(1, 9),
                    torch.randn(1, 4, 7 if sample["task"] == "t2v" else 1, 16, 24, generator=rng), False)

        def reconstruction(encoder, sample):
            rng = torch.Generator().manual_seed(700 + sample["micro"])
            return [torch.randn(1, 13, 8, generator=rng) for _ in range(3)], torch.ones(1, 13)

        trainer.load_stack = stack
        trainer.verify_release = lambda root: None
        media.plan_from_release = lambda *a, **kw: media.MediaEpochPlan(17, 19, 2, 1,
            epochs=kw["epochs"], allow_partial_cycle=True, rng_world_size=1)
        media.MediaEpochDataset = Dataset
        trainer.encode_sample = encode
        trainer.validate = lambda *a: {}
        trainer.stage2.reconstruction_condition = reconstruction
        torch.cuda.max_memory_allocated = lambda device: 0
        ddp_class = torch.nn.parallel.DistributedDataParallel
        trainer.DDP = lambda module, **kw: ddp_class(module, **{k: v for k, v in kw.items() if k != "device_ids"})
        out = directory / ("migrated" if mode == "resume" else "fresh_ddp")
        if rank == 0:
            out.mkdir(exist_ok=True)
        dist.barrier()
        common = ["--phase", "2", "--output-dir", str(out), "--config", str(directory / "config"),
                  "--data-root", str(directory / "data"), "--workers", "0", "--accumulation", "1",
                  "--media-epochs", "5", "--extend-media-epochs", "--repartition-media-batch",
                  "--media-rng-world-size", "1", "--reconstruction-weight", ".1",
                  "--save-every", "1000", "--archive-every", "1000", "--validate-every", "1000", "--log-every", "1"]
        ctx = DistributedContext(rank, rank, 2, torch.device("cpu"))
        if mode == "history":
            common.remove("--extend-media-epochs")
            common.remove("--repartition-media-batch")
            common.extend(["--init-connector", str(directory / "init.pt"), "--fresh-media-history",
                           "--history-corrupt-max", str(1/3)])
            trainer.train(trainer.parse_args([*common, "--stop-after", "7"]), ctx)
            trainer.train(trainer.parse_args([*common, "--resume", str(out / "stage2_resume.pt")]), ctx)
            return
        # Pause after the first update, then resume the new layout with its own RNG/Adam.
        trainer.train(trainer.parse_args([*common, "--resume", str(directory / "parent/stage2_resume.pt"),
                                          "--stop-after", "58"]), ctx)
        trainer.train(trainer.parse_args([*common, "--resume", str(out / "stage2_resume.pt")]), ctx)
    finally:
        dist.destroy_process_group()


def test_actual_trainer_ddp_repartition_and_own_resume(training_environment):
    env = training_environment
    env.run("parent")
    reference = env.run("reference", resume=env.root / "parent/stage2_resume.pt", epochs=5)
    mp.spawn(migrated_trainer_worker, args=(str(env.root),), nprocs=2, join=True)
    actual = torch.load(env.root / "migrated/stage2_resume.pt", weights_only=True)
    expected = torch.load(reference / "stage2_resume.pt", weights_only=True)
    assert actual["step"] == expected["step"] == 95
    assert actual["contract"]["world_size"] == 2 and actual["contract"]["accumulation"] == 1
    assert len(actual["rng_states"]) == 2
    for component in ("connector", "dit"):
        for key, value in actual[component].items():
            torch.testing.assert_close(value, expected[component][key], atol=1e-5, rtol=1e-4)
    for actual_moment, expected_moment in zip(actual["optimizer"]["state"].values(), expected["optimizer"]["state"].values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_moment[key], expected_moment[key], atol=1e-5, rtol=1e-4)
    report = json.loads((env.root / "migrated/resume_from_step000057.json").read_text())
    assert report["epoch_extension"]["ddp_repartitioned"] and report["epoch_extension"]["max_adam_step"] == 57
    resumed = json.loads((env.root / "migrated/resume_from_step000058.json").read_text())
    assert not resumed["epoch_extension"]["ddp_repartitioned"] and resumed["epoch_extension"]["global_rng_restored"]


def test_actual_trainer_fresh_history_ddp_accumulation_one(training_environment):
    env = training_environment
    mp.spawn(migrated_trainer_worker, args=(str(env.root), "history"), nprocs=2, join=True)
    result = env.root / "fresh_ddp"
    saved = torch.load(result / "stage2_resume.pt", weights_only=True)
    assert saved["step"] == 95 and len(saved["rng_states"]) == 2
    assert saved["contract"]["initialization"] == "stage1_connector_and_released_multistep_dit"
    assert saved["contract"]["accumulation"] == 1
    assert saved["contract"]["auxiliary_reconstruction"]["weight"] == .1
    audit = json.loads((result / "history_corruption_audit.json").read_text())
    assert audit["step"] <= 7 and audit["actual"]["changed_rms"] > 0
    for epoch in (3, 4, 5):
        assert (result / f"stage2_dit_connector_step{19 * epoch:06d}.pt").exists()


def test_fresh_history_from_alignment_saves_epochs_and_resumes_exactly(training_environment, monkeypatch):
    env = training_environment
    original_plan = env.plan
    monkeypatch.setattr(media, "plan_from_release", lambda *a, **kw: media.MediaEpochPlan(17, 19, 1, 2,
        epochs=kw["epochs"], allow_partial_cycle=True, rng_world_size=1))
    def run(name, extra=()):
        out = env.root / name
        out.mkdir(exist_ok=True)
        args = trainer.parse_args([*fresh_cli(env.root / "init.pt", out),
            "--config", str(env.root / "config"), "--data-root", str(env.data), "--workers", "0",
            "--media-rng-world-size", "1", "--save-every", "1000", "--archive-every", "1000",
            "--validate-every", "1000", "--log-every", "1", "--verify-frozen", *extra])
        trainer.train(args, DistributedContext(0, 0, 1, torch.device("cpu")))
        return out
    full, paused = run("history_full"), run("history_paused", ["--stop-after", "7"])
    run("history_paused", ["--resume", str(paused / "stage2_resume.pt")])
    a, b = [torch.load(root / "stage2_resume.pt", weights_only=True) for root in (full, paused)]
    same(a, b)
    assert a["contract"]["initialization"] == "stage1_connector_and_released_multistep_dit"
    assert a["contract"]["history_corruption"]["max_sigma"] == 1/3
    assert "first_frame_focus" not in a["contract"]
    audit = json.loads((full / "history_corruption_audit.json").read_text())
    assert audit["actual"]["changed_rms"] > 0 and audit["step"] < original_plan().steps
    for epoch in (1, 2, 3, 4, 5):
        step = epoch * original_plan().steps_per_epoch
        assert (full / f"stage2_dit_connector_step{step:06d}.pt").exists()


def test_two_node_shell_launch_and_safe_continuation(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    source, init = tmp_path / "stage2_resume.pt", tmp_path / "init.pt"
    source.write_text("optimizer")
    init.write_text("alignment15k")
    out = tmp_path / "out"
    out.mkdir()
    commands = {
        "module": "exit 0",
        "python": 'printf "PLAN %s\\n" "$*" >> "$CALLS"; exit "${PREFLIGHT_RC:-0}"',
        "scontrol": 'printf "node01\\nnode02\\n"',
        "srun": '''printf "SRUN %s\\n" "$*" >> "$CALLS"
if [[ "${TRAIN_RC:-0}" != 0 ]]; then exit "$TRAIN_RC"; fi
SLURM_NODEID=0 bash scripts/run_mobileov_stage2_2node16gpu.sh --node
SLURM_NODEID=1 bash scripts/run_mobileov_stage2_2node16gpu.sh --node''',
        "torchrun": 'printf "TRAIN %s\\n" "$*" >> "$CALLS"',
        "sbatch": 'printf "CONTINUE %s RESUME=%s OUT=%s\\n" "$*" "$RESUME" "$OUT" >> "$CALLS"; echo 999',
    }
    for name, body in commands.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", CONDA_ENV=str(tmp_path),
               PROJECT_ROOT=str(ROOT), SLURM_NNODES="2", SLURM_JOB_NODELIST="node[01-02]",
               SLURM_JOB_ID="123", TMPDIR=str(tmp_path / "tmp"), OUT=str(out), CALLS=str(calls))
    for key in ("RESUME", "STEPS", "LR", "ACCUMULATION", "INIT_STAGE2", "INIT_CONNECTOR", "AUTO_CONTINUE"):
        env.pop(key, None)
    def run(job, **updates):
        calls.unlink(missing_ok=True)
        return subprocess.run(["bash", str(job)], env={**env, **updates}, capture_output=True, text=True)
    for job in (RESUME_JOB, HISTORY_JOB, RUNNER):
        subprocess.run(["bash", "-n", str(job)], check=True)
    assert run(RESUME_JOB, RESUME=str(source)).returncode == 0
    result = calls.read_text()
    assert "--world-size 16 --accumulation 1 --media-rng-world-size 8 --repartition-media-batch" in result
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1" in result
    assert "--node_rank=0" in result and "--node_rank=1" in result
    assert result.count("TRAIN ") == 2 and "--standalone" not in result
    assert "--resume" in result and "--init-connector" not in result
    assert run(HISTORY_JOB, INIT_CONNECTOR=str(init)).returncode == 0
    result = calls.read_text()
    assert "--fresh-media-history --history-corrupt-max 0.3333333333333333" in result
    assert "--media-epochs 5" in result and "--init-stage2" not in result
    assert "--validation-samples 1" in result
    for job, args in ((RESUME_JOB, dict(RESUME=str(source))), (HISTORY_JOB, dict(INIT_CONNECTOR=str(init)))):
        assert run(job, **args, PREFLIGHT_RC="1").returncode != 0
        assert "TRAIN" not in calls.read_text() and "CONTINUE" not in calls.read_text()
        assert run(job, **args, TRAIN_RC="1").returncode != 0
        assert "CONTINUE" not in calls.read_text()
        (out / "training_paused.json").write_text("{}")
        (out / "stage2_resume.pt").write_text("own_optimizer")
        assert run(job, **args).returncode == 0
        result = calls.read_text()
        assert f"RESUME={out}/stage2_resume.pt OUT={out}" in result
        assert "--dependency=afterok:123" in result
    assert run(RESUME_JOB, RESUME=str(source), LR="1e-4").returncode != 0
    assert run(RESUME_JOB, RESUME=str(source), SLURM_NNODES="1").returncode != 0
    assert run(HISTORY_JOB, INIT_CONNECTOR=str(init), RESUME=str(source)).returncode != 0
