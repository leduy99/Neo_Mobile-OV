#!/usr/bin/env python
"""Actual SmolVLM2 CPU gradient/export check; not a NeoDragon GPU quality test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch

from new_mobile_ov.bridge.stage1_conditioner import FrozenStage1Encoder, Stage1Connector
from new_mobile_ov.training import job_b_model as job
from new_mobile_ov.training.job_b_qa import pack_answer
from new_mobile_ov.training.stage2_alignment import gradient_norm
from tools.data_prepare.download_alignment_images import atomic_json


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    frozen = FrozenStage1Encoder(args.checkpoint, args.processor, torch.device("cpu"), torch.float32)
    before, mask = frozen("A red toy car driving on a road.")
    encoder = job.LoRAEncoder(frozen)
    connector = Stage1Connector(input_dim=encoder.input_dim)
    after, _ = encoder("A red toy car driving on a road.")
    if any(not torch.equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError("Zero-initialized LoRA changed the original conditioner")
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([*parameters, *connector.parameters()], lr=1e-5, weight_decay=0)
    versions = [(p, p._version) for p in encoder.parameters() if not p.requires_grad]
    original_adapters = job.lora_state(encoder)
    tokens, _, pooled = connector(after, mask)
    (tokens.square().mean() + pooled.square().mean()).backward()
    feature_gradient = float(gradient_norm(encoder))
    connector_gradient = float(gradient_norm(connector))
    if min(feature_gradient, connector_gradient) <= 0:
        raise RuntimeError("Actual MCP conditioning probe did not reach LLM LoRA and connector")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    for modality, images in (("text", []), ("image", [Image.new("RGB", (512, 320), "red")]),
                             ("video", [Image.new("RGB", (512, 320), "red") for _ in range(8)])):
        row = dict(sample_id=f"smoke:{modality}", modality=modality,
                   media=[dict(path="synthetic", sha256="unused", seconds=i) for i in range(len(images))],
                   messages=[dict(role="user", content="What color is shown?" if images else "What is two plus two?"),
                             dict(role="assistant", content="Red." if images else "Four.")])
        batch = pack_answer(encoder.processor, row, images)
        loss = encoder.qa_loss(batch)
        (.1 * loss).backward()
        norm = float(gradient_norm(encoder))
        if norm <= 0 or any(p.grad is not None for p, _ in versions):
            raise RuntimeError(f"Invalid actual {modality} QA gradient path")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics[modality] = dict(answer_ce=float(loss.detach()), lora_gradient_norm=norm,
                                 tokens=int(batch["inputs"]["input_ids"].shape[1]),
                                 answer_tokens=int(batch["answer_mask"].sum()))
        del batch, loss
    if any(p._version != version for p, version in versions):
        raise RuntimeError("Frozen SmolVLM2 parameters changed")
    state = job.lora_state(encoder)
    if not any(not torch.equal(state[key], original_adapters[key]) for key in state):
        raise RuntimeError("LoRA optimizer did not update actual weights")
    encoder.eval()
    with torch.no_grad():
        output, _ = encoder("A red toy car driving on a road.")
        for parameter in parameters:
            parameter.add_(.01)
    job.load_lora(encoder, state)
    reloaded, _ = encoder("A red toy car driving on a road.")
    if any(not torch.equal(a, b) for a, b in zip(output, reloaded)):
        raise RuntimeError("Inference adapter export/reload changed conditioner output")
    report = dict(status="actual_SmolVLM2_MCP_QA_CPU_mechanics_passed", initial_conditioner_preserved_exactly=True,
                  lora_parameters=sum(p.numel() for p in parameters), feature_probe_gradient=feature_gradient,
                  connector_probe_gradient=connector_gradient, qa=metrics,
                  frozen_parameters_unchanged=True, inference_reload_exact=True,
                  real_NeoDragon_DiT_GPU_tested=False, QA_accuracy_evaluated=False, quality_claim="none")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/smolvlm2_500m/smolvlm2_500m.pt"))
    parser.add_argument("--processor", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("output/job_b_encoder_smoke/actual_encoder_check.json"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("Threads must be in [1,16]")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
