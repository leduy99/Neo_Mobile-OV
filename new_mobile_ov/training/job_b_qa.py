"""Deterministic image/text/video answer replay for Job B, separate from captions."""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from new_mobile_ov.training.stage1_alignment import JsonlRecords, permuted_index
from tools.data_prepare.download_alignment_images import sha256_file

FORMAT = "mobileov_job_b_qa_v1"
MODALITIES = ("image", "text", "video")
REPLAY_CYCLE = ("image",) * 10 + ("text",) * 4 + ("video",) * 6


def verify_release(root):
    root = Path(root)
    report = json.loads((root / "qa_summary.json").read_text())
    if report.get("format") != FORMAT or report.get("status") != "mechanically_verified_QA_release":
        raise ValueError("Prepare the Job-B QA release; raw download completion is not sufficient")
    index_path = root / "media_index.json"
    if sha256_file(index_path) != report.get("media_index_sha256"):
        raise ValueError("QA media index checksum differs")
    for reference in json.loads(index_path.read_text()):
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != reference["sha256"]:
            raise ValueError("Missing, corrupt or unsafe prepared QA media")
    for split in ("train", "validation"):
        for modality in MODALITIES:
            name = f"{split}/{modality}.jsonl"
            path = root / name
            if sha256_file(path) != report["files"][name]["sha256"]:
                raise ValueError(f"QA manifest checksum differs: {name}")
            if len(JsonlRecords(path)) != report["files"][name]["rows"] or report["files"][name]["rows"] < 1:
                raise ValueError(f"QA manifest is empty or its row count differs: {name}")
    return report


def read_images(root, row):
    images = []
    for reference in row.get("media", []):
        path = (Path(root) / reference["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("QA media escapes release root")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise ValueError("QA media checksum differs")
        with Image.open(io.BytesIO(payload)) as image:
            images.append(image.convert("RGB"))
    expected = 0 if row["modality"] == "text" else 1 if row["modality"] == "image" else 8
    if len(images) != expected:
        raise ValueError("QA modality/media count differs")
    return images


def chat_messages(row):
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant" or len(messages) < 2:
        raise ValueError("QA requires a final assistant answer")
    result, inserted = [], False
    for message in messages:
        role, text = message["role"], message["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(text, str) or not text.strip():
            raise ValueError("Unsupported or empty QA message")
        if "<image>" in text or "<video>" in text:
            raise ValueError("QA placeholders must be normalized before training")
        content = []
        if role == "user" and not inserted:
            if row["modality"] == "video":
                content.append(dict(type="text", text="These frames are in chronological order."))
            for reference in row.get("media", []):
                if "seconds" in reference:
                    content.append(dict(type="text", text=f"Frame at {reference['seconds']:.2f} seconds:"))
                content.append(dict(type="image"))
            inserted = True
        content.append(dict(type="text", text=text))
        result.append(dict(role=role, content=content))
    if not inserted:
        raise ValueError("QA conversation contains no user message")
    return result


def pack_answer(processor, row, images, max_tokens=2048):
    messages = chat_messages(row)
    prefix = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    kwargs = dict(return_tensors="pt", padding=False, truncation=False)
    if images:
        kwargs.update(images=[images], images_kwargs={"do_image_splitting": False})
    prompt = processor(text=[prefix], **kwargs)
    encoded = processor(text=[full], **kwargs)
    ids, prefix_ids = encoded["input_ids"], prompt["input_ids"]
    length = prefix_ids.shape[1]
    if ids.shape[0] != 1 or not 0 < length < ids.shape[1] <= max_tokens:
        raise ValueError("QA exceeds token budget or contains no answer; no silent truncation")
    if not torch.equal(ids[:, :length], prefix_ids):
        raise ValueError("Processor changed the assistant prefix; answer mask would be incorrect")
    if images and "pixel_values" not in encoded:
        raise ValueError("QA images did not reach the vision input")
    inputs = {key: value for key, value in encoded.items()
              if key in {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"}}
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, length:] = True
    return dict(inputs=inputs, answer_mask=mask, sample_id=row["sample_id"], modality=row["modality"])


def answer_ce(hidden, head, batch, chunk_tokens=64):
    ids, attention, mask = batch["inputs"]["input_ids"], batch["inputs"]["attention_mask"], batch["answer_mask"]
    if hidden.shape[:2] != ids.shape or mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError("QA hidden states/mask do not match tokens")
    if bool(mask[:, 0].any()) or bool((mask & ~attention.bool()).any()):
        raise ValueError("Cannot supervise position zero or padded QA tokens")
    valid = mask[:, 1:] & attention[:, 1:].bool() & attention[:, :-1].bool()
    states, targets = hidden[:, :-1][valid], ids[:, 1:][valid]
    if len(targets) < 1 or any(p.requires_grad for p in head.parameters()):
        raise ValueError("Answer CE requires answer tokens and a frozen LM head")
    loss = hidden.new_zeros((), dtype=torch.float32)
    def project(value, labels):
        return F.cross_entropy(head(value).float(), labels, reduction="sum")
    for offset in range(0, len(targets), chunk_tokens):
        values = states[offset:offset + chunk_tokens], targets[offset:offset + chunk_tokens]
        loss = loss + (checkpoint(project, *values, use_reentrant=False)
                       if torch.is_grad_enabled() and states.requires_grad else project(*values))
    return loss / len(targets)


class Replay:
    def __init__(self, root, *, processor, seed, rank, world_size, max_tokens=2048):
        self.root, self.processor = Path(root), processor
        self.seed, self.rank, self.world_size, self.max_tokens = seed, rank, world_size, max_tokens
        self.records = {key: JsonlRecords(self.root / "train" / f"{key}.jsonl") for key in MODALITIES}

    def sample(self, ordinal):
        presentation = ordinal * self.world_size + self.rank
        modality = REPLAY_CYCLE[presentation % len(REPLAY_CYCLE)]
        records = self.records[modality]
        row = records[permuted_index(presentation, len(records), self.seed + 777777, modality)]
        return pack_answer(self.processor, row, read_images(self.root, row), self.max_tokens)
