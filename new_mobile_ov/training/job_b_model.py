"""One distributed owner for LLM LoRA, MCP, DiT and answer-only CE; no KL."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from new_mobile_ov.bridge.stage1_conditioner import FrozenStage1Encoder
from new_mobile_ov.training.job_b_qa import answer_ce
from new_mobile_ov.training.stage1_alignment import flow_loss, frozen_signatures

FORMAT = "mobileov_job_b_lora_mcp_dit_v1"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("Invalid LoRA projection/rank")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid LoRA alpha")
        self.base, self.scale = base.requires_grad_(False), alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value):
        original = self.base(value)
        update = F.linear(F.linear(value.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return original + update.to(original.dtype) * self.scale


def inject_lora(tower, rank=16, alpha=32):
    if any(isinstance(module, LoRALinear) for module in tower.modules()):
        raise ValueError("LoRA was already installed")
    selected = [(name, module) for name, module in tower.named_modules()
                if name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, nn.Linear)]
    if {name.rsplit(".", 1)[-1] for name, _ in selected} != set(TARGETS):
        raise ValueError("Unexpected text-tower architecture; missing LoRA targets")
    replacements = [(name, LoRALinear(module, rank, alpha)) for name, module in selected]
    tower.requires_grad_(False)
    for name, module in replacements:
        parent, _, leaf = name.rpartition(".")
        setattr(tower.get_submodule(parent) if parent else tower, leaf, module)
    return [name for name, _ in selected]


def lora_state(encoder):
    return {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()
            if name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}}


def load_lora(encoder, state):
    expected = lora_state(encoder)
    if not expected or set(expected) != set(state):
        raise ValueError("LoRA checkpoint keys differ")
    if any(state[key].shape != expected[key].shape or state[key].dtype != torch.float32
           or not bool(torch.isfinite(state[key]).all()) for key in state):
        raise ValueError("Invalid LoRA checkpoint tensors")
    encoder.load_state_dict(state, strict=False)


class LoRAEncoder(nn.Module):
    def __init__(self, frozen, *, rank=16, alpha=32, gradient_checkpointing=True):
        super().__init__()
        self.backbone = frozen.backbone
        self.processor, self.processor_sha256 = frozen.processor, frozen.processor_sha256
        self.max_tokens, self.input_dim = frozen.max_tokens, frozen.input_dim
        self.backbone.requires_grad_(False)
        self.lora_spec = dict(rank=rank, alpha=alpha, dropout=0, targets=list(TARGETS))
        self.installed = inject_lora(self.backbone._model.model.text_model, rank, alpha)
        if gradient_checkpointing:
            self.backbone._model.model.text_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # SmolVLMModel's wrapper enables gradients only when its own training flag is true.
        self.backbone._model.eval()
        self.backbone._model.model.text_model.train(mode)
        self.backbone.training = mode
        return self

    def forward(self, prompt, image=None, **kwargs):
        return FrozenStage1Encoder.forward.__wrapped__(self, prompt, image, **kwargs)

    def qa_loss(self, batch):
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in batch["inputs"].items()}
        batch = dict(batch, inputs=inputs, answer_mask=batch["answer_mask"].to(device))
        result = self.backbone(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        return answer_ce(result.last_hidden_state, self.backbone._model.lm_head, batch)


class JointModel(nn.Module):
    def __init__(self, encoder, connector, dit):
        super().__init__()
        self.encoder, self.connector, self.dit = encoder, connector, dit

    def forward(self, *, qa=None, prompt=None, image=None, drop=False, allow_text_image=False,
                latent=None, scheduler=None, stage=None, unit=None, generator=None):
        if qa is not None:
            return self.encoder.qa_loss(qa)
        layers, mask = self.encoder(prompt, image, drop_condition=drop, allow_text_image=allow_text_image)
        return flow_loss(self.dit, self.connector, layers, mask, latent, scheduler, stage=stage,
                         unit=unit, generator=generator, gradient_checkpointing=False)


def load_inference(payload, frozen_encoder, connector, dit):
    """Load ALL three trained parts; never silently infer with missing LLM adapters."""
    if payload.get("format") != FORMAT or payload.get("connector_spec") != connector.spec:
        raise ValueError("Invalid Job-B inference checkpoint/connector")
    contract = payload["contract"]
    if (frozen_encoder.processor_sha256 != contract["processor_sha256"]
            or frozen_signatures(dict(smolvlm2=frozen_encoder))["smolvlm2"] != contract["frozen_weights_sha256"]["smolvlm2"]):
        raise ValueError("Job-B inference requires the same frozen SmolVLM2 base and processor")
    spec = payload["lora_spec"]
    if spec != contract["lora_spec"] or spec["dropout"] != 0 or spec["targets"] != list(TARGETS):
        raise ValueError("Job-B inference LoRA spec differs")
    encoder = LoRAEncoder(frozen_encoder, rank=spec["rank"], alpha=spec["alpha"], gradient_checkpointing=False)
    load_lora(encoder, payload["smolvlm2_lora"])
    connector.float().load_state_dict(payload["connector"], strict=True)
    dit.float().load_state_dict(payload["dit"], strict=True)
    return JointModel(encoder, connector, dit).eval().requires_grad_(False)
