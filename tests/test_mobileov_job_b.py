from contextlib import nullcontext
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq

from new_mobile_ov.training import job_b_model as job
from new_mobile_ov.training import job_b_qa as qa
from tools import prepare_mobileov_job_b_qa as prepare
from tools import train_mobileov_job_b as trainer
from new_mobile_ov.bridge.stage1_conditioner import Stage1Connector
from new_mobile_ov.training.distributed import DistributedContext
from new_mobile_ov.training.stage1_alignment import frozen_signatures
from new_mobile_ov.training.stage2_media_epochs import MediaEpochPlan
from tools.data_prepare.download_alignment_images import sha256_file
from test_stage1_alignment_data import encoded_video


class Processor:
    chat_template = "fixture"
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 1})
    image_processor = SimpleNamespace(to_dict=lambda: {"size": 512})

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = ""
        for message in messages:
            result += message["role"] + ":"
            result += "".join(x.get("text", "[image]") for x in message["content"]) + "|"
        return result + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[ord(c) % 128 for c in text[0]]])
        result = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        if kwargs.get("images"):
            result["pixel_values"] = torch.ones(1, 1, 3, 8, 8)
        return result


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 32)
        for name in job.TARGETS:
            setattr(self, name, nn.Linear(32, 32))

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing_kwargs = kwargs

    def forward(self, ids):
        embedding = self.embedding(ids)
        hidden = embedding
        for name in job.TARGETS:
            hidden = torch.tanh(getattr(self, name)(hidden))
        return embedding, hidden, hidden * .9


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Module()
        self._model.model = nn.Module()
        self._model.model.text_model = Tower()
        self._model.model.vision = nn.Linear(3, 32)
        self._model.lm_head = nn.Linear(32, 128)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.enable_grad() if self.training else torch.no_grad():
            states = self._model.model.text_model(input_ids)
            return SimpleNamespace(last_hidden_state=states[-1], hidden_states=states)


def encoder():
    frozen = SimpleNamespace(backbone=Backbone().requires_grad_(False).eval(), processor=Processor(),
                             processor_sha256="proc", max_tokens=2048, input_dim=32)
    return job.LoRAEncoder(frozen)


def qa_row(modality="text", media=None):
    return dict(sample_id="example", modality=modality, media=media or [], group_id="group",
                messages=[dict(role="user", content="What is two plus two?"),
                          dict(role="assistant", content="Four.")])


def test_answer_mask_prefix_and_shift():
    batch = qa.pack_answer(Processor(), qa_row(), [])
    ids, mask = batch["inputs"]["input_ids"], batch["answer_mask"]
    assert int(mask.sum()) == len("Four.|") and not mask[:, 0].any()
    hidden = torch.randn(1, ids.shape[1], 8, requires_grad=True)
    head = nn.Linear(8, 128).requires_grad_(False)
    expected = F.cross_entropy(head(hidden[:, :-1])[mask[:, 1:]], ids[:, 1:][mask[:, 1:]])
    actual = qa.answer_ce(hidden, head, batch, chunk_tokens=2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert hidden.grad.norm() > 0 and head.weight.grad is None
    with pytest.raises(ValueError, match="token budget"):
        qa.pack_answer(Processor(), qa_row(), [], max_tokens=10)


def test_lora_zero_init_gradient_frozen_weights_and_export_reload():
    enc = encoder()
    ids = torch.randint(128, (1, 12))
    tower = enc.backbone._model.model.text_model
    versions = [(p, p._version) for p in enc.parameters() if not p.requires_grad]
    state = copy.deepcopy(enc.state_dict())
    adapted = enc.backbone(input_ids=ids).last_hidden_state
    assert adapted.requires_grad and all(p.dtype == torch.float32 for p in enc.parameters() if p.requires_grad)
    adapted.square().mean().backward()
    assert sum(float(p.grad.norm()) for p in enc.parameters() if p.requires_grad and p.grad is not None) > 0
    optimizer = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    assert all(p.grad is None and p._version == version for p, version in versions)
    assert any(not torch.equal(value, state[key]) for key, value in enc.state_dict().items() if "lora_B" in key)
    exported = job.lora_state(enc)
    other = encoder()
    other.load_state_dict(state)
    job.load_lora(other, exported)
    torch.testing.assert_close(enc.backbone(input_ids=ids).last_hidden_state,
                               other.backbone(input_ids=ids).last_hidden_state)
    assert tower.checkpointing_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    enc.eval()
    assert not enc.backbone(input_ids=ids).last_hidden_state.requires_grad


def test_qa_and_generation_reach_same_lora_encoder():
    enc = encoder()
    layers, mask = enc("a short prompt")
    assert layers[-1].requires_grad and mask.all()
    layers[-1].square().mean().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in enc.parameters() if p.requires_grad)
    enc.zero_grad(set_to_none=True)
    batch = qa.pack_answer(enc.processor, qa_row(), [])
    enc.qa_loss(batch).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in enc.parameters() if p.requires_grad)
    assert all(p.grad is None for p in enc.parameters() if not p.requires_grad)


def test_replay_ratio_and_resume_are_deterministic(tmp_path):
    (tmp_path / "train").mkdir()
    for modality in qa.MODALITIES:
        rows = [dict(qa_row(modality), sample_id=f"{modality}:{i}") for i in range(100)]
        (tmp_path / "train" / f"{modality}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    seen = []
    # Test sampler scheduling independently of media decoding.
    import unittest.mock as mock
    with mock.patch.object(qa, "read_images", return_value=[]), mock.patch.object(qa, "pack_answer", side_effect=lambda p, r, i, n: r):
        for rank in range(2):
            replay = qa.Replay(tmp_path, processor=Processor(), seed=42, rank=rank, world_size=2)
            seen.extend(replay.sample(i)["modality"] for i in range(20))
            assert replay.sample(13) == qa.Replay(tmp_path, processor=Processor(), seed=42, rank=rank, world_size=2).sample(13)
    assert {key: seen.count(key) for key in qa.MODALITIES} == {"image": 20, "text": 8, "video": 12}


def test_conversation_prefixes_and_media_bounds(tmp_path):
    messages = [{"from": "human", "value": "<image>\nQuestion?"}, {"from": "gpt", "value": "Answer."},
                {"from": "human", "value": "Next?"}, {"from": "gpt", "value": "Next answer."}]
    rows = list(prepare.prefixes(messages))
    assert len(rows) == 2 and rows[0][0]["content"] == "Question?"
    reference = prepare.put_image(Image.new("RGB", (1024, 512)), tmp_path)
    row = qa_row("image", [reference])
    assert qa.read_images(tmp_path, row)[0].size == (512, 256)
    (tmp_path / reference["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        qa.read_images(tmp_path, row)


def test_job_b_parser_does_not_silently_enable_kl_or_change_job_a():
    flags = ["--qa-root", "qa", "--phase", "2", "--init-stage2", "parent.pt", "--output-dir", "out",
             "--media-epochs", "1", "--media-start-epoch", "3", "--t2v-first-probability", ".5", "--reconstruction-weight", ".2"]
    args = trainer.parse_args(flags)
    assert args.qa_every == 8 and args.qa_weight == .1 and args.lora_lr == 1e-5
    assert args.lr == 2e-5 and args.lora_warmup == 1000 and args.accumulation == 2
    for extra in (["--qa-weight", "nan"], ["--lora-lr", "1e-3"], ["--qa-every", "0"]):
        with pytest.raises(SystemExit):
            trainer.parse_args([*flags, *extra])


class TinyOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = job.LoRALinear(nn.Linear(8, 8), rank=2, alpha=4)
        self.connector, self.dit = nn.Linear(8, 4), nn.Linear(4, 2)

    def forward(self, x, qa_only=False):
        hidden = self.adapter(x)
        return hidden.square().mean() if qa_only else self.dit(self.connector(hidden)).square().mean()


def ddp_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=rank, world_size=2)
    try:
        torch.manual_seed(10)
        owner = TinyOwner()
        model = torch.nn.parallel.DistributedDataParallel(owner, find_unused_parameters=True)
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 20
        with model.no_sync():
            (.1 * model(x[rank], qa_only=True)).backward()
        for micro in range(2):
            with model.no_sync() if micro == 0 else nullcontext():
                (model(x[rank] + micro) / 2).backward()
        if rank == 0:
            torch.save({k: p.grad for k, p in owner.named_parameters() if p.requires_grad}, Path(root) / "grads.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Needs CPU Gloo")
def test_qa_before_generation_ddp_accumulation_matches_global_batch(tmp_path):
    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    torch.manual_seed(10)
    owner = TinyOwner()
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 20
    for rank in range(2):
        (.05 * owner(x[rank], qa_only=True)).backward()
        for micro in range(2):
            (owner(x[rank] + micro) / 4).backward()
    observed = torch.load(tmp_path / "grads.pt", weights_only=True)
    for name, parameter in owner.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(observed[name], parameter.grad, rtol=1e-5, atol=1e-7)


@pytest.fixture
def prepared_qa(tmp_path):
    source, root = tmp_path / "sources", tmp_path / "qa"
    entries = []
    def add(path, source_name, kind):
        entries.append(dict(filename=str(path.relative_to(source / "sources" / source_name)),
                            source=source_name, size=path.stat().st_size, hash=sha256_file(path),
                            hash_algorithm="sha256", kind=kind))
    images = []
    for index in range(40):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (index * 5, 40, 80)).save(buf, "PNG")
        images.append(dict(images=[dict(bytes=buf.getvalue(), path=None)],
                           texts=[dict(user=f"What is shown in image {index}?", assistant="An image.")]))
    path = source / "sources/image_qa/vqa/train-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(images), path)
    add(path, "image_qa", "train_parquet")
    path = source / "sources/text_instruction/data/train-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(messages=[dict(role="user", content=f"Instruction {i}?"),
                                                       dict(role="assistant", content="An answer.")]) for i in range(40)]), path)
    add(path, "text_instruction", "train_parquet")
    path = source / "sources/video_qa/videos/media.tar.gz"
    path.parent.mkdir(parents=True)
    annotations = []
    with tarfile.open(path, "w:gz") as archive:
        for index in range(20):
            name = f"academic_source/fixture/{index}.mp4"
            payload = encoded_video(frames=8 + index)
            member = tarfile.TarInfo(f"0_30_s_academic_v0_1/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
            annotations.append(dict(video=name, conversations=[{"from": "human", "value": f"<image>\nVideo {index}?"},
                                                               {"from": "gpt", "value": "A changing light."}]))
    add(path, "video_qa", "media_archive")
    path = path.parent / "qa.json"
    path.write_text(json.dumps(annotations))
    add(path, "video_qa", "qa_annotation")
    (source / "source_files_complete.json").write_text(json.dumps(dict(
        status="source_files_verified_QA_release_NOT_prepared", plan_sha256="a" * 64, files=entries)))
    args = SimpleNamespace(source_root=source, output_dir=root, processor="fixture", max_tokens=2048,
                           workers=2, seed=42, validation_fraction=.2)
    report = prepare.build(args, Processor())
    return root, args, report


def test_preparation_joins_media_checks_every_answer_and_disjoint_split(prepared_qa):
    root, args, report = prepared_qa
    assert report == qa.verify_release(root)
    assert report == prepare.build(args, Processor())
    assert sum(value["rows"] for value in report["files"].values()) == 100
    groups = {}
    for split in ("train", "validation"):
        groups[split] = set()
        for modality in qa.MODALITIES:
            records = qa.JsonlRecords(root / split / f"{modality}.jsonl")
            for i in range(len(records)):
                row = records[i]
                groups[split].add(row["group_id"])
                qa.pack_answer(Processor(), row, qa.read_images(root, row))
    assert not groups["train"] & groups["validation"]
    args.seed = 7
    with pytest.raises(ValueError, match="contract changed"):
        prepare.build(args, Processor())


@pytest.mark.skipif(os.environ.get("RUN_ACTUAL_JOB_B_PROCESSOR") != "1", reason="Optional real HF processor check")
def test_actual_processor_preparation_is_thread_safe_and_prefix_exact(prepared_qa):
    from transformers import AutoProcessor
    root, args, _ = prepared_qa
    args = copy.copy(args)
    args.output_dir = root.parent / "actual_processor_qa"
    args.workers = 8
    args.processor = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(args.processor)
    report = prepare.build(args, processor)
    assert report == qa.verify_release(args.output_dir)
    for modality in qa.MODALITIES:
        row = qa.JsonlRecords(args.output_dir / "train" / f"{modality}.jsonl")[0]
        batch = qa.pack_answer(processor, row, qa.read_images(args.output_dir, row))
        assert int(batch["answer_mask"].sum()) > 0
    assert sum(value["rows"] for value in report["files"].values()) == 100


def test_production_runner_fork_save_and_exact_resume_on_tiny_stack(tmp_path, prepared_qa, monkeypatch):
    qa_root, _, qa_report = prepared_qa
    data_root = tmp_path / "generation"
    data_root.mkdir()
    (data_root / "stage1_summary.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text("fixture")
    plan = MediaEpochPlan(4, 4, 1, 2, epochs=1, start_epoch=3, seed=20260911)
    def stack(args, device):
        frozen = SimpleNamespace(backbone=Backbone().requires_grad_(False).eval(), processor=Processor(),
                                 processor_sha256=qa_report["processor_sha256"], max_tokens=2048, input_dim=32)
        # Signatures operate on nn.Modules like the real frozen encoder.
        frozen_module = nn.Module()
        frozen_module.backbone = frozen.backbone
        for key in ("processor", "processor_sha256", "max_tokens", "input_dim"):
            setattr(frozen_module, key, getattr(frozen, key))
        return None, frozen_module, Stage1Connector(32, 16, 12), nn.Linear(12, 1), nn.Linear(1, 1).requires_grad_(False), None
    torch.manual_seed(7)
    _, frozen, connector, dit, vae, _ = stack(None, None)
    original = dict(format="mobileov_stage2_mcp_dit_v1", config_sha256=sha256_file(config), processor="fixture",
                    processor_sha256=qa_report["processor_sha256"],
                    frozen_weights_sha256=frozen_signatures(dict(smolvlm2=frozen, vae=vae)),
                    short_side=320, long_side=512, max_tokens=2048, condition_dropout=.1, vae_window_size=8,
                    geometry="nearest_aspect_bucket_resize_full_view_no_crop",
                    temporal="49_distinct_frames_over_whole_source_clip; normalized_duration_not_native_fps",
                    steps=12, media_epochs=MediaEpochPlan(4, 4, 1, 2).contract(),
                    data_summary_sha256=sha256_file(data_root / "stage1_summary.json"))
    parent = tmp_path / "parent.pt"
    torch.save(dict(format=original["format"], step=12, contract=original, connector_spec=connector.spec,
                    connector=connector.state_dict(), dit=dit.state_dict()), parent)
    monkeypatch.setattr(trainer.base, "load_stack", stack)
    monkeypatch.setattr(trainer.media, "plan_from_release", lambda *a, **k: plan)
    monkeypatch.setattr(trainer, "validate", lambda *a, **k: {})
    monkeypatch.setattr(trainer, "target_latent", lambda vae, sample, *a, **k:
                        torch.zeros(1, 1, 7 if sample["task"] == "t2v" else 1, 8, 8))
    def fake_flow(dit, connector, layers, mask, latent, scheduler, **kwargs):
        _, _, pooled = connector(layers, mask)
        return (dit(pooled).float() - .3).square().mean()
    monkeypatch.setattr(job, "flow_loss", fake_flow)
    class Dataset(torch.utils.data.Dataset):
        sources = {}
        def __init__(self, root, plan, rank, start_step, **kwargs):
            self.start = start_step
        def __len__(self):
            return (4 - self.start) * 2
        def __getitem__(self, index):
            step, micro = divmod(index, 2)
            slot = plan.slot(step + self.start, micro, 0)
            return dict(slot, task=slot["source"], sample_id=f"{slot['source']}:{slot['index']}",
                        prompt="A small light.", image=None, video=torch.zeros(3, 1, 64, 64))
    monkeypatch.setattr(trainer.media, "MediaEpochDataset", Dataset)
    ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
    def arguments(out, end, resume=None):
        argv = ["--qa-root", str(qa_root), "--qa-every", "1", "--phase", "2", "--init-stage2", str(parent),
                "--expected-stage2-step", "12", "--output-dir", str(out), "--config", str(config),
                "--processor", "fixture", "--data-root", str(data_root), "--workers", "0",
                "--media-epochs", "1", "--media-start-epoch", "3", "--t2v-first-probability", ".5",
                "--reconstruction-weight", ".2", "--integration-smoke", "--stop-after", str(end),
                "--save-every", "2", "--archive-every", "2"]
        if resume:
            argv += ["--resume", str(resume)]
        return trainer.parse_args(argv)
    full, continued = tmp_path / "full", tmp_path / "continued"
    for out, end, resume in ((full, 16, None), (continued, 14, None), (continued, 16, continued / "job_b_resume.pt")):
        out.mkdir(exist_ok=True)
        torch.manual_seed(7)
        trainer.train(arguments(out, end, resume), ctx)
    a = torch.load(full / "job_b_dit_connector_lora_latest.pt", weights_only=False)
    b = torch.load(continued / "job_b_dit_connector_lora_latest.pt", weights_only=False)
    for name in ("connector", "dit", "smolvlm2_lora"):
        for key in a[name]:
            torch.testing.assert_close(a[name][key], b[name][key], rtol=0, atol=0)
    assert "optimizer" not in a and "optimizer" in torch.load(continued / "job_b_resume.pt", weights_only=False)
    assert a["step"] == b["step"] == 16
    assert (continued / "first_qa_gradient_audit.json").exists()
    torch.manual_seed(7)
    _, frozen, connector, dit, _, _ = stack(None, None)
    loaded = job.load_inference(a, frozen, connector, dit)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    for key, value in job.lora_state(loaded.encoder).items():
        torch.testing.assert_close(value, a["smolvlm2_lora"][key], rtol=0, atol=0)


def test_job_b_scripts_syntax_and_dependency(tmp_path):
    root = Path(__file__).resolve().parents[1]
    names = ("train_mobileov_stage2_job_b_lora_1node8gpu.sbatch", "submit_mobileov_job_b_pair.sh",
             "smoke_mobileov_job_b_local.sbatch")
    for name in names:
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "sbatch"
    mock.write_text('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n'
                    'p=Path(os.environ["CAPTURE"]); rows=json.loads(p.read_text()) if p.exists() else []\n'
                    'rows.append(sys.argv[1:]); p.write_text(json.dumps(rows)); print(100+len(rows))\n')
    mock.chmod(0o755)
    capture = tmp_path / "calls.json"
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", CAPTURE=str(capture))
    subprocess.run(["bash", str(root / "scripts/submit_mobileov_job_b_pair.sh")], env=env, check=True)
    calls = json.loads(capture.read_text())
    assert len(calls) == 2
    assert calls[0][-1] == "scripts/download_mobileov_job_b_data_1node1gpu.sbatch"
    assert "--dependency=afterok:101" in calls[1] and "--kill-on-invalid-dep=yes" in calls[1]
    assert calls[1][-1] == "scripts/train_mobileov_stage2_job_b_lora_1node8gpu.sbatch"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_training_launcher_uses_8_gpus_and_correct_parent_recipe(tmp_path, exit_code):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/train_mobileov_stage2_job_b_lora_1node8gpu.sbatch"
    binary = tmp_path / "env/bin"
    binary.mkdir(parents=True)
    module = binary / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    python = binary / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    torchrun = binary / "torchrun"
    torchrun.write_text('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n'
                        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:])); '
                        'sys.exit(int(os.environ["EXIT_CODE"]))\n')
    torchrun.chmod(0o755)
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    (qa_root / "qa_summary.json").write_text("{}")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"fixture")
    capture = tmp_path / "argv.json"
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", PROJECT_ROOT=str(tmp_path),
               CONDA_ENV=str(binary.parent), QA_ROOT=str(qa_root), INIT_STAGE2=str(parent),
               OUT=str(tmp_path / "out"), TMPDIR=str(tmp_path / "tmp"), SLURM_JOB_ID="123",
               CAPTURE=str(capture), EXIT_CODE=str(exit_code), AUTO_CONTINUE="0")
    env.pop("STEPS", None)
    env.pop("INIT_CONNECTOR", None)
    env.pop("RESUME", None)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    argv = json.loads(capture.read_text())
    assert "--nproc_per_node=8" in argv and "tools/train_mobileov_job_b.py" in argv
    assert argv[argv.index("--expected-stage2-step") + 1] == "283977"
    assert argv[argv.index("--qa-root") + 1] == str(qa_root)
    assert argv[argv.index("--qa-every") + 1] == "8"
    assert argv[argv.index("--lora-lr") + 1] == "1e-5"
    assert argv[argv.index("--lora-warmup") + 1] == "1000"
