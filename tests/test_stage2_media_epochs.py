from collections import Counter, defaultdict
from contextlib import nullcontext
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

from new_mobile_ov.training import stage2_media_epochs as media
from tools import train_mobileov_stage1 as trainer
from test_stage1_training import fake_encoder
from test_stage2_training import TinyDiT, scheduler, install_history, batch
from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training.stage2_alignment import JointFlowModel


@pytest.mark.parametrize("images,videos,world,acc", [(1, 1, 8, 2), (11, 7, 2, 2), (16, 32, 8, 2), (3, 5, 2, 3)])
def test_exact_passes_all_modes_weights_padding_and_rank_collectives(images, videos, world, acc):
    plan = media.MediaEpochPlan(images, videos, world, acc)
    counts, modes, weights = Counter(), defaultdict(Counter), Counter()
    for step in range(plan.steps):
        sources = set()
        for micro in range(acc):
            for rank in range(world):
                slot = plan.slot(step, micro, rank)
                sources.add(slot["source"])
                if slot["padding"]:
                    assert slot["loss_scale"] == 0
                    continue
                counts[slot["source"], slot["index"], slot["epoch"]] += 1
                modes[slot["source"], slot["index"]][slot["mode"]] += 1
                weights[slot["source"], slot["epoch"]] += slot["loss_scale"] / plan.global_batch / plan.steps_per_epoch
        assert len(sources) == 1  # Every rank executes the same number of reconstruction backwards.
    assert len(counts) == (images + videos) * 3 and set(counts.values()) == {1}
    for i in range(videos):
        assert modes["t2v", i] == Counter(dict.fromkeys(media.VIDEO_MODES, 1))
    for i in range(images):
        assert modes["t2i", i] == {"t2i": 3}
    assert all(value == pytest.approx(.5) for value in weights.values())


def test_expected_budget_and_resume_order():
    plan = media.MediaEpochPlan(984636, 531238, 8, 2)
    assert plan.steps == 284229
    assert plan.contract()["paired_reconstruction_presentations"] == 984636 * 3
    assert plan.contract()["presentations_per_video_mode"] == 531238
    other = media.MediaEpochPlan(984636, 531238, 8, 2)
    for step in (0, 10000, plan.steps_per_epoch - 1, plan.steps_per_epoch, plan.steps - 1):
        for rank in range(8):
            assert plan.slot(step, 1, rank) == other.slot(step, 1, rank)
    with pytest.raises(ValueError):
        media.MediaEpochPlan(2, 3, 2, 2, epochs=2)


def test_image_text_route_is_opt_in_and_drops_both():
    encoder, image = fake_encoder(), object()
    with pytest.raises(ValueError, match="caption"):
        encoder("move left", image)
    encoder("move left", image, allow_text_image=True)
    assert encoder.processor.messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "move left"}]
    assert "pixel_values" in encoder.backbone.inputs
    encoder("move left", image, allow_text_image=True, drop_condition=True)
    assert encoder.processor.messages[0]["content"] == [{"type": "text", "text": ""}]
    assert "pixel_values" not in encoder.backbone.inputs


def test_only_first_frame_reaches_vlm_and_anchor_never_supervises_unit0():
    video = torch.ones(3, 49, 8, 8)
    video[:, 0] = -1
    sample = dict(task="t2v", prompt="moving", image=None, video=video)
    for mode in media.VIDEO_MODES:
        conditioned = media.condition_video(sample, mode)
        assert conditioned["prompt"] == "moving"
        if mode == "t2v":
            assert conditioned["image"] is None
        else:
            assert conditioned["image"].getpixel((0, 0)) == (0, 0, 0)
        rng = torch.Generator().manual_seed(8)
        units = {media.sample_unit(mode, 7, "cpu", rng) for _ in range(100)}
        assert units == set(range(1 if mode == "i2v_anchor" else 0, 7))
    assert sample["image"] is None


def test_anchor_encoding_is_independent_first_image_even_with_dropout(monkeypatch):
    inputs = []
    def posterior(vae, video, **kwargs):
        inputs.append(video.clone())
        value = 9 if video.shape[2] == 1 else 4
        return SimpleNamespace(sample=lambda generator: torch.full((1, 4, 1 if value == 9 else 7, 8, 8), value))
    monkeypatch.setattr(trainer, "encode_posterior", posterior)
    monkeypatch.setattr(trainer, "scale_vae_latents", lambda x: x)
    encoder = fake_encoder()
    video = torch.zeros(3, 49, 64, 64)
    video[:, 1:] = 1
    sample = media.condition_video(dict(task="t2v", prompt="move", video=video), "i2v_anchor")
    _, _, latent, dropped = trainer.encode_sample(encoder, None, sample, torch.device("cpu"),
                                                  torch.Generator().manual_seed(1), 1, window_size=8)
    assert dropped and "pixel_values" not in encoder.backbone.inputs
    assert [v.shape[2] for v in inputs] == [49, 1] and not inputs[1].any()
    assert bool((latent[:, :, :1] == 9).all()) and bool((latent[:, :, 1:] == 4).all())


def test_data_guards_and_cli(tmp_path):
    (tmp_path / "train").mkdir()
    for task, count in (("t2i", 3), ("t2v", 5)):
        (tmp_path / "train" / f"{task}.jsonl").write_text("{}\n" * count)
    (tmp_path / "stage1_summary.json").write_text(json.dumps(dict(counts={"train.t2i": 3, "train.t2v": 5})))
    kwargs = dict(world_size=2, accumulation=2, verify=False)
    with pytest.raises(ValueError, match="Resume data preparation"):
        media.plan_from_release(tmp_path, require_expansion=True, **kwargs)
    with pytest.raises(ValueError, match="old subset"):
        media.plan_from_release(tmp_path, min_videos=500000, **kwargs)
    assert media.plan_from_release(tmp_path, **kwargs).steps == 9
    (tmp_path / "expansion_report.json").write_text(json.dumps(dict(status="complete", total_train_videos=5)))
    with pytest.raises(ValueError, match="disagree"):
        media.plan_from_release(tmp_path, require_expansion=True, **kwargs)
    (tmp_path / "stage1_summary.json").write_text(json.dumps(dict(
        counts={"train.t2i": 3, "train.t2v": 5}, expansion=dict(status="complete", total_train_videos=5))))
    assert media.plan_from_release(tmp_path, require_expansion=True, **kwargs).steps == 9
    base = ["--phase", "2", "--init-connector", "init.pt", "--output-dir", "unused", "--media-epochs", "3",
            "--reconstruction-weight", ".1"]
    assert trainer.parse_args(base).media_epochs == 3
    for extra in (["--steps", "100000"], ["--extend-steps"], ["--reconstruction-weight", "0"]):
        with pytest.raises(SystemExit):
            trainer.parse_args([*base, *extra])
    ctx = SimpleNamespace(device="cpu", is_distributed=False)
    means, counts = media.distributed_loss_means({"t2i": [torch.tensor(2.)]}, ctx)
    assert means["t2i"] == 2 and means["i2v_anchor"] is None and counts["i2v_anchor"] == 0


def test_budget_cli_persists_plan_and_rejects_changes(tmp_path, monkeypatch):
    from tools import plan_mobileov_stage2_epochs as planner
    from tools.data_prepare.download_alignment_images import sha256_file
    (tmp_path / "stage1_summary.json").write_text("verified_release")
    plan = media.MediaEpochPlan(3, 5, 8, 2)
    monkeypatch.setattr(planner, "plan_from_release", lambda *a, **kw: plan)
    output = tmp_path / "run/epoch_plan.json"
    monkeypatch.setattr(sys, "argv", ["planner", "--data-root", str(tmp_path), "--output", str(output)])
    planner.main()
    saved = json.loads(output.read_text())
    assert saved["data_summary_sha256"] == sha256_file(tmp_path / "stage1_summary.json")
    assert saved["steps"] == plan.steps and saved["paired_reconstruction_presentations"] == 9
    planner.main()
    (tmp_path / "stage1_summary.json").write_text("another_release")
    with pytest.raises(ValueError, match="Existing training budget differs"):
        planner.main()
    assert json.loads(output.read_text()) == saved


def model_batch(model, plan, step, rank):
    for micro in range(plan.accumulation):
        slot = plan.slot(step, micro, rank)
        is_image = slot["source"] == "t2i"
        # Deliberately different token lengths and modes on the two ranks.
        data = batch(rank, step, 0 if is_image else 1)
        unit = 0 if is_image else (1 if slot["mode"] == "i2v_anchor" else 6)
        yield slot, micro, data, unit


def ddp_epoch_worker(rank, directory):
    torch.set_num_threads(1)
    install_history()
    dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        module = JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
        ddp = torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True,
                                                       gradient_as_bucket_view=True, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=2e-5)
        plan = media.MediaEpochPlan(3, 5, 2, 2)
        for step in range(plan.steps):
            optimizer.zero_grad()
            for slot, micro, data, unit in model_batch(ddp, plan, step, rank):
                with ddp.no_sync() if micro == 0 else nullcontext():
                    loss = ddp(*data, scheduler(), stage=step % 3, unit=unit,
                               generator=torch.Generator().manual_seed(step + micro))
                    (loss * slot["loss_scale"] / 2).backward()
                    if slot["source"] == "t2i":
                        rec = ddp(*data, scheduler(), stage=2, unit=0,
                                  generator=torch.Generator().manual_seed(step + micro + 100))
                        (rec * .2 * slot["loss_scale"] / 2).backward()
            optimizer.step()
        torch.save(module.state_dict(), Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_source_grouped_ddp_matches_global_objective_including_padding(tmp_path, monkeypatch):
    mp.spawn(ddp_epoch_worker, args=(str(tmp_path),), nprocs=2, join=True)
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    torch.manual_seed(42)
    model = JointFlowModel(Stage1Connector(8, 16, 12), TinyDiT())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    plan = media.MediaEpochPlan(3, 5, 2, 2)
    for step in range(plan.steps):
        optimizer.zero_grad()
        for rank in range(2):
            for slot, micro, data, unit in model_batch(model, plan, step, rank):
                loss = model(*data, scheduler(), stage=step % 3, unit=unit,
                             generator=torch.Generator().manual_seed(step + micro))
                (loss * slot["loss_scale"] / 4).backward()
                if slot["source"] == "t2i":
                    rec = model(*data, scheduler(), stage=2, unit=0,
                                generator=torch.Generator().manual_seed(step + micro + 100))
                    (rec * .2 * slot["loss_scale"] / 4).backward()
        optimizer.step()
    a = torch.load(tmp_path / "rank0.pt", weights_only=True)
    b = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, value in model.state_dict().items():
        assert torch.equal(a[key], b[key])
        assert torch.allclose(a[key], value, atol=1e-6, rtol=1e-4), key


def test_shell_syntax():
    root = Path(__file__).resolve().parents[1]
    for name in ("train_mobileov_stage2_media3epochs_1node8gpu.sbatch", "smoke_mobileov_stage2_media_epochs_local.sbatch"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    script = (root / "scripts/train_mobileov_stage2_media3epochs_1node8gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:8" in script and "#SBATCH --cpus-per-task=128" in script
    assert "--save-every 10000 --archive-every 10000" in script
    assert "--reconstruction-weight 0.1" in script


def test_new_training_loop_exact_resume_and_old_checkpoint_rejected(tmp_path, monkeypatch):
    from new_mobile_ov.training.distributed import DistributedContext
    from new_mobile_ov.training.stage1_alignment import frozen_signatures
    from tools.data_prepare.download_alignment_images import sha256_file
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", SimpleNamespace(
        _prepare_past_condition_latents=lambda past, stages, cfg: [past] * stages))
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage1_summary.json").write_text("{}")
    config, smol = tmp_path / "config", tmp_path / "smol"
    config.write_text("config")
    smol.write_text("smol")
    cfg = SimpleNamespace(bridge=SimpleNamespace(smolvlm2_ckpt_path=str(smol)))
    plan = media.MediaEpochPlan(3, 5, 1, 2)

    def stack(args, device):
        encoder, vae, dit = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), TinyDiT()
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
    monkeypatch.setattr(trainer, "load_stack", stack)
    monkeypatch.setattr(trainer, "verify_release", lambda root: None)
    monkeypatch.setattr(media, "plan_from_release", lambda *args, **kwargs: plan)
    monkeypatch.setattr(media, "MediaEpochDataset", Dataset)
    monkeypatch.setattr(trainer, "encode_sample", encode)
    monkeypatch.setattr(trainer, "validate", lambda *args: {})
    monkeypatch.setattr(trainer.stage2, "reconstruction_condition", reconstruction)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    torch.manual_seed(plan.seed)
    _, encoder, connector, dit, vae, _ = stack(None, None)
    init = tmp_path / "init.pt"
    torch.save(dict(format=trainer.FORMAT, step=15000, connector_spec=connector.spec,
                    connector=connector.state_dict(), contract=dict(config_sha256=sha256_file(config),
                    processor_sha256="processor", data_summary_sha256="original_release",
                    frozen_weights_sha256=frozen_signatures(dict(smolvlm2=encoder, dit=dit, vae=vae)),
                    short_side=320, long_side=512, max_tokens=2048)), init)
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    def run(name, extra=()):
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        args = trainer.parse_args(["--phase", "2", "--init-connector", str(init), "--config", str(config),
            "--data-root", str(data), "--output-dir", str(out), "--media-epochs", "3",
            "--reconstruction-weight", ".1", "--workers", "0", "--log-every", "1", "--verify-frozen", *extra])
        trainer.train(args, ctx)
        return out
    full = run("full")
    paused = run("paused", ["--stop-after", "4"])
    run("paused", ["--resume", str(paused / "stage2_resume.pt")])
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
    a = torch.load(full / "stage2_resume.pt", weights_only=True)
    b = torch.load(paused / "stage2_resume.pt", weights_only=True)
    same(a, b)
    assert a["step"] == plan.steps and a["contract"]["media_epochs"] == plan.contract()
    rows = [json.loads(line) for line in (paused / "history.jsonl").read_text().splitlines()]
    seen = Counter((sample["task"], sample["sample_id"]) for row in rows for sample in row["rank0_samples"]
                   if not sample["padding"])
    for index in range(5):
        for mode in media.VIDEO_MODES:
            assert seen[mode, f"t2v:{index}"] == 1
    for index in range(3):
        assert seen["t2i", f"t2i:{index}"] == seen["image_reconstruction", f"t2i:{index}"] == 3
    old = dict(b, contract={key: value for key, value in b["contract"].items() if key != "media_epochs"})
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        trainer.validate_resume(old, b["contract"], b["connector_spec"])


def test_launcher_derived_budget_preflight_and_safe_continuation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    out = tmp_path / "out"
    out.mkdir()
    init = tmp_path / "init.pt"
    init.write_text("checkpoint")
    env = dict(os.environ, PROJECT_ROOT=str(root), SLURM_JOB_ID="12345", OUT=str(out),
               INIT_CONNECTOR=str(init), DATA_ROOT=str(tmp_path / "data"), CONDA_ENV=str(tmp_path),
               PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls), TMPDIR=str(tmp_path / "tmp"))
    for key in ("STEPS", "RESUME", "ACCUMULATION", "AUTO_CONTINUE"):
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
    script = root / "scripts/train_mobileov_stage2_media3epochs_1node8gpu.sbatch"
    def run(**updates):
        return subprocess.run(["bash", str(script)], env={**env, **updates}, capture_output=True, text=True)
    assert run().returncode == 0
    text = calls.read_text()
    assert "--media-epochs 3" in text and "--steps " not in text
    assert "--require-expanded-release" in text and "--min-train-videos 500000" in text
    assert f"--output {out}/epoch_plan.json" in text
    assert f"--init-connector {init} --expected-init-step 15000" in text
    assert "CONTINUE" not in text
    (out / "training_paused.json").write_text("{}")
    (out / "stage2_resume.pt").write_text("checkpoint")
    assert run().returncode == 0
    assert "CONTINUE --parsable --dependency=afterok:12345 --export=ALL" in calls.read_text()
    assert f"RESUME={out}/stage2_resume.pt OUT={out}" in calls.read_text()
    assert (out / "continuation_jobs.txt").read_text() == "12346\n"
    calls.unlink()
    assert run(AUTO_CONTINUE="0").returncode == 0
    assert "CONTINUE" not in calls.read_text()
    calls.unlink()
    assert run(PREFLIGHT_RC="1").returncode != 0
    assert "TRAIN " not in calls.read_text() and "CONTINUE" not in calls.read_text()
    calls.unlink()
    assert run(TRAIN_RC="1").returncode != 0
    assert "CONTINUE" not in calls.read_text()
    assert run(STEPS="100000").returncode != 0
    assert run(AUTO_CONTINUE="maybe").returncode != 0
    assert run(INIT_CONNECTOR=str(tmp_path / "missing.pt")).returncode != 0
    (out / "run_contract.json").write_text("{}")
    assert run().returncode != 0
    assert run(RESUME=str(out / "stage2_resume.pt"), AUTO_CONTINUE="0").returncode == 0


def test_inference_anchor_preserves_unit0_and_generates_six_future_units(monkeypatch):
    from tools import infer_mobileov_stage1 as infer
    calls, decoded = [], []
    def generate(scheduler, dit, stages, noise, history, *condition, **kwargs):
        calls.append((len(history), kwargs["num_inference_steps"], history[0].clone() if history else None))
        return noise + 1
    utils = SimpleNamespace(
        _prepare_latent_noise=lambda batch, channels, units, h, w, *args: torch.zeros(batch, channels, units, h, w),
        _downsample_noise_2x=lambda latent, times: latent,
        _prepare_past_condition_latents=lambda generated, stages, cfg: generated.copy(),
        _generate_one_unit=generate,
        _decode_latent=lambda vae, latent: decoded.append(latent.clone()) or [None] * 49)
    monkeypatch.setitem(sys.modules, "neodragon.utils.generation_utils", utils)
    monkeypatch.setattr(infer, "install_neodragon_generation_patches", lambda **kwargs: None)
    monkeypatch.setattr(infer, "autocast", lambda device: nullcontext())
    anchor = torch.full((1, 4, 1, 8, 8), 7.)
    dit = SimpleNamespace(config=SimpleNamespace(in_channels=4))
    kwargs = dict(device=torch.device("cpu"), seed=1, height=64, width=64)
    _, stats = infer.generate_frames(dit, None, None, (), anchor_latent=anchor, **kwargs)
    assert len(calls) == 6 and [c[0] for c in calls] == list(range(1, 7))
    assert all(c[1] == [10] * 3 and torch.equal(c[2], anchor) for c in calls)
    assert torch.equal(decoded[0][:, :, :1], anchor) and stats[0]["supplied_anchor"]
    calls.clear()
    infer.generate_frames(dit, None, None, (), **kwargs)
    assert len(calls) == 7 and calls[0][:2] == (0, [20] * 3)
    with pytest.raises(ValueError, match="Invalid video anchor"):
        infer.generate_frames(dit, None, None, (), anchor_latent=anchor[:, :, :, :, :4], **kwargs)
    args = SimpleNamespace(first_frame=Path("frame.png"), reconstruct_image=None, validation_task=None,
                           frames=49, prompt="move left", i2v_mode="anchor")
    with pytest.raises(ValueError, match="not trained"):
        infer.verify_i2v_request(args, {})
    infer.verify_i2v_request(args, dict(media_epochs=media.MediaEpochPlan(3, 5, 2, 2).contract()))


def test_dataset_suffix_after_resume_and_provenance(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "sources.json").write_text("{}")
    plan = media.MediaEpochPlan(3, 5, 2, 2)
    for source, count in (("t2i", 3), ("t2v", 5)):
        (tmp_path / "train" / f"{source}.jsonl").write_text("".join(
            json.dumps(dict(task=source, sample_id=f"{source}:{i}")) + "\n" for i in range(count)))
    monkeypatch.setattr(media, "prepare_sample", lambda record, *args: dict(record))
    monkeypatch.setattr(media, "condition_video", lambda sample, mode: dict(sample, mode=mode))
    for rank in range(2):
        full = media.MediaEpochDataset(tmp_path, plan=plan, rank=rank)
        suffix = media.MediaEpochDataset(tmp_path, plan=plan, rank=rank, start_step=4)
        assert [suffix[i] for i in range(len(suffix))] == [full[i] for i in range(8, len(full))]
