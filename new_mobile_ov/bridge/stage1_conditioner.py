"""Stage-1 multimodal MCP alignment, separate from the legacy 128-token distiller."""
from __future__ import annotations

import hashlib
import json
import torch
from torch import nn

from new_mobile_ov.bridge.sana_prompt_bridge import MCPProjector
from new_mobile_ov.bridge.text_bridge import pool_prompt_tokens


class Stage1Connector(nn.Module):
    def __init__(self, input_dim=960, token_dim=1536, pooled_dim=2048, fuse_layers=2):
        super().__init__()
        self.spec = dict(input_dim=input_dim, token_dim=token_dim,
                         pooled_dim=pooled_dim, fuse_layers=fuse_layers)
        self.projector = MCPProjector(input_dim, token_dim, num_fuse_layers=fuse_layers,
                                      use_refine=True, lexical_mode="gated_add", lexical_gate_init=0.2)
        self.pooled_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, pooled_dim))

    def forward(self, layers, mask):
        if mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("Stage-1 uses one unpadded sample per GPU microbatch")
        if not bool(mask.all()) or any(h.shape[:2] != mask.shape for h in layers):
            raise ValueError("Invalid feature/mask contract; do not silently select 128 tokens")
        tokens = self.projector(layers)
        pooled = self.pooled_head(pool_prompt_tokens(tokens, mask))
        return tokens, mask, pooled


class FrozenStage1Encoder(nn.Module):
    def __init__(self, checkpoint, processor_id, device, dtype, max_tokens=2048):
        super().__init__()
        from transformers import AutoProcessor
        from new_mobile_ov.smolvlm2 import load_smolvlm2_from_ckpt, SmolVLMModel

        self.backbone = load_smolvlm2_from_ckpt(checkpoint, device=device, model_class=SmolVLMModel)
        self.backbone.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        processor_state = dict(vocabulary=self.processor.tokenizer.get_vocab(),
                               chat_template=self.processor.chat_template,
                               image_processor=self.processor.image_processor.to_dict())
        self.processor_sha256 = hashlib.sha256(json.dumps(
            processor_state, sort_keys=True, default=str).encode()).hexdigest()
        self.max_tokens = max_tokens
        config = self.backbone._model.config
        image_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_id != config.image_token_id:
            raise ValueError("Processor image token differs from SmolVLM2 checkpoint")
        self.input_dim = config.text_config.hidden_size

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, prompt, image=None, *, drop_condition=False):
        if drop_condition:
            prompt, image = "", None
        if image is not None and prompt:
            raise ValueError("Reconstruction is image-only; target caption must not leak into the MLLM")
        content = [{"type": "image"}] if image is not None else [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kwargs = dict(text=[text], return_tensors="pt", padding=False, truncation=False)
        if image is not None:
            kwargs.update(images=[[image]], images_kwargs={"do_image_splitting": False})
        encoded = self.processor(**kwargs)
        if encoded["input_ids"].shape[1] > self.max_tokens:
            raise ValueError(f"Input exceeds {self.max_tokens} tokens; refusing silent caption/image truncation")
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in encoded.items()
                  if key in {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"}}
        if image is not None and "pixel_values" not in inputs:
            raise RuntimeError("Image reconstruction did not reach the vision encoder")
        outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        if not outputs.hidden_states:
            raise RuntimeError("Missing SmolVLM2 hidden layers")
        # MCP consumes the embedding layer and the last two decoder layers only.
        return [outputs.hidden_states[0], *outputs.hidden_states[-2:]], inputs["attention_mask"]
