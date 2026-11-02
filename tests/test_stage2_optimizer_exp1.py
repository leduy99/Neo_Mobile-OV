import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from new_mobile_ov.training import stage2_optimizer_exp1 as exp1
from new_mobile_ov.training import stage2_media_epochs as media
from new_mobile_ov.training.distributed import DistributedContext
from tools import train_mobileov_stage1 as legacy
from tools import train_mobileov_stage2_optimizer_exp1 as trainer
from test_stage2_epoch_extension_resume import training_environment, same


def test_schedule_no_rewarm_and_zero_final_lr():
    assert exp1.learning_rate(0) == 2e-5
    assert exp1.learning_rate(10000) == 1e-5
    assert exp1.learning_rate(20000) == 0
    assert exp1.learning_rate(5, total=10, decay_steps=2) == 2e-5
    assert exp1.learning_rate(9, total=10, decay_steps=2) == 1e-5
    for update in (-1, 20001):
        with pytest.raises(ValueError):
            exp1.learning_rate(update)


def test_fp32_ema_math_restore_and_swap_exception():
    model = torch.nn.Linear(2, 1).float()
    modules = dict(dit=model)
    ema = exp1.FP32EMA(modules, .9)
    before = exp1.cpu_weights(model)
    with torch.no_grad():
        model.weight.add_(2)
        model.bias.sub_(1)
    raw = exp1.cpu_weights(model)
    ema.update(modules)
    assert ema.updates == 1
    assert torch.allclose(ema.shadow["dit"]["weight"], before["weight"] + .2)
    assert torch.allclose(ema.shadow["dit"]["bias"], before["bias"] - .1)
    fresh = exp1.FP32EMA(modules, .9)
    fresh.load_state_dict(ema.state_dict(), expected_updates=1)
    same(fresh.state_dict(), ema.state_dict())
    with pytest.raises(ValueError):
        fresh.load_state_dict(ema.state_dict(), expected_updates=2)
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    with pytest.raises(RuntimeError):
        with exp1.ema_weights(modules, ema, ctx):
            same(model.state_dict(), ema.shadow["dit"])
            raise RuntimeError("validation failed")
    same(model.state_dict(), raw)


def test_grouped_batch64_preserves_exact_original_media_and_noise_ids(monkeypatch):
    old = media.MediaEpochPlan(984636, 529902, 8, 2)
    parent = dict(media_epochs=old.contract(), steps=old.steps)

    class Slots:
        def __init__(self, root, *, plan, rank, start_step, **kw):
            self.plan, self.rank, self.start = plan, rank, start_step
            self.sources, self.records = {}, {}
        def __getitem__(self, index):
            step, micro = divmod(index, self.plan.accumulation)
            return self.plan.slot(self.start + step, micro, self.rank)
    monkeypatch.setattr(media, "MediaEpochDataset", Slots)
    plan = exp1.parent_plan(parent)
    assert plan.steps == 378636
    for update in (0, 9999, 19999):
        for rank in range(16):
            ds = exp1.GroupedMediaDataset("unused", parent=parent, rank=rank, world_size=16, start_update=update)
            assert len(ds) == (20000 - update) * 4
            for micro in range(4):
                slot = ds[micro]
                expected = plan.slot(old.steps + update * 4 + micro, rank // 8, rank % 8)
                assert slot == dict(expected, rng_rank=rank % 8)
    assert (old.steps + 20000 * 4) / old.steps_per_epoch == pytest.approx(3.8451399233)


def run_exp1(env, name, extra=()):
    output = env.root / name
    output.mkdir(exist_ok=True)
    args = trainer.parse_args(["--parent-resume", str(env.root / "parent/stage2_resume.pt"),
        "--output-dir", str(output), "--data-root", str(env.data), "--config", str(env.root / "config"),
        "--integration-smoke", "--expected-parent-step", "108", "--steps", "6", "--decay-steps", "6",
        "--workers", "0", "--log-every", "1", "--save-every", "2", "--evaluate-every", "3",
        "--validate-every", "2", *extra])
    trainer.train(args, DistributedContext(0, 0, 1, torch.device("cpu")))
    return output


def prepare_parent(env, monkeypatch):
    monkeypatch.setattr(media.MediaEpochDataset, "sources", {}, raising=False)
    monkeypatch.setattr(media, "plan_from_release", lambda *a, epochs=3, **kw:
        media.MediaEpochPlan(17, 19, 1, 1, epochs=epochs, allow_partial_cycle=True))
    return env.run("parent", extra=["--accumulation", "1"])


def test_actual_trainer_restores_adam_ema_schedule_and_rng_exactly(training_environment, monkeypatch):
    env = training_environment
    env.stochastic["enabled"] = True
    parent = prepare_parent(env, monkeypatch)
    original = (parent / "stage2_resume.pt").read_bytes()
    def noisy_validation(*a):
        torch.rand(17)
        return {"validation": 1.}
    monkeypatch.setattr(legacy, "validate", noisy_validation)
    full = run_exp1(env, "exp1_full")
    paused = run_exp1(env, "exp1_paused", ["--stop-after", "2"])
    run_exp1(env, "exp1_paused", ["--resume", str(paused / "exp1_resume.pt")])
    read = lambda out: torch.load(out / "exp1_resume.pt", weights_only=True)
    a, b = read(full), read(paused)
    same(a, b)
    assert a["step"] == 114 and a["exp1_update"] == a["ema"]["updates"] == 6
    assert a["sampler_cursor"] == 108 + 6 * 4
    assert a["optimizer"]["param_groups"][0]["lr"] == 0
    assert a["scheduler"]["completed_updates"] == 6
    assert (parent / "stage2_resume.pt").read_bytes() == original
    source = torch.load(parent / "stage2_resume.pt", weights_only=True)
    first = json.loads((full / "first_update_optimizer_audit_update000001.json").read_text())
    assert first["max_adam_step"] == 109 and first["ema_updates"] == 1 and not first["warmup_restarted"]
    assert max(value["step"].item() for value in source["optimizer"]["state"].values()) == 108
    manifest = json.loads((full / "evaluation_checkpoints.json").read_text())
    assert set(manifest["checkpoints"]) == {"raw_3", "ema_3", "raw_6", "ema_6"}
    latest = torch.load(full / "stage2_dit_connector_latest.pt", weights_only=True)
    assert latest["weight_variant"] == "ema" and "optimizer" not in latest and "ema" not in latest
    same(latest["dit"], a["ema"]["shadow"]["dit"])
    assert any(not torch.equal(latest["dit"][key], a["dit"][key]) for key in latest["dit"])
    rows = [json.loads(line) for line in (full / "history.jsonl").read_text().splitlines()]
    assert [r["lr"] for r in rows] == [exp1.learning_rate(n, total=6, decay_steps=6) for n in range(1, 7)]
    assert all(sum(r["valid_samples_in_update"][k] for k in media.VIDEO_MODES) + r["valid_samples_in_update"]["t2i"] <= 4 for r in rows)


def test_reject_inference_reset_wrong_source_and_missing_ema(training_environment, monkeypatch):
    env = training_environment
    parent = prepare_parent(env, monkeypatch)
    with pytest.raises(ValueError, match="Missing original Adam"):
        run_exp1(env, "inference", ["--parent-resume", str(parent / "stage2_dit_connector_latest.pt")])
    source = torch.load(parent / "stage2_resume.pt", weights_only=True)
    bad = copy.deepcopy(source)
    for state in bad["optimizer"]["state"].values():
        state["step"] = torch.tensor(1.)
    path = env.root / "reset.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="counters"):
        run_exp1(env, "reset", ["--parent-resume", str(path)])
    paused = run_exp1(env, "paused", ["--stop-after", "2"])
    saved = torch.load(paused / "exp1_resume.pt", weights_only=True)
    for field in ("ema", "scheduler", "sampler_cursor", "rng_states"):
        candidate = copy.deepcopy(saved)
        candidate.pop(field)
        torch.save(candidate, paused / "exp1_resume.pt")
        with pytest.raises(ValueError):
            run_exp1(env, "paused", ["--resume", str(paused / "exp1_resume.pt")])
    torch.save(saved, paused / "exp1_resume.pt")
    with pytest.raises(ValueError, match="Existing Exp-1"):
        run_exp1(env, "paused")


def test_production_cli_is_explicit_and_pinned():
    args = trainer.parse_args(["--output-dir", "unused"])
    assert (args.steps, args.decay_steps, args.ema_decay, args.accumulation) == (20000, 20000, .9995, 4)
    for extra in (["--accumulation", "1"], ["--steps", "100000"], ["--ema-decay", "nan"],
                  ["--decay-steps", "10000"], ["--expected-parent-step", "473295"]):
        with pytest.raises(SystemExit):
            trainer.parse_args(["--output-dir", "unused", *extra])


def test_new_contract_retains_learned_i2v_capabilities_without_fake_epoch_count():
    from tools.infer_mobileov_stage1 import verify_i2v_request
    original = media.MediaEpochPlan(984636, 529902, 8, 2)
    parent = dict(steps=original.steps, media_epochs=original.contract())
    contract = exp1.continuation_contract(parent, parent_sha256="a" * 64, world_size=16, accumulation=4,
        total=20000, decay_steps=20000, ema_decay=.9995)
    args = SimpleNamespace(first_frame=Path("image.png"), reconstruct_image=None, validation_task=None,
                           frames=49, prompt="A car moves.", i2v_mode="anchor")
    verify_i2v_request(args, contract)
    assert contract["media_epochs"]["global_batch"] == 64 and "epochs" not in contract["media_epochs"]
    assert contract["media_epochs"]["end_media_cursor"] == 363977
