from __future__ import annotations

import math
from typing import List, Literal, NamedTuple, Sequence

import torch
import torch.nn as nn
from PIL import Image

from new_mobile_ov.bridge.ssd1b_image_bridge import ImageQueryBlock, _detect_hidden_size
from new_mobile_ov.bridge.sana_prompt_bridge import SimpleTokenizer
from new_mobile_ov.checkpoints import ensure_smolvlm2_checkpoint
from new_mobile_ov.config import BridgeConfig, DreamLiteBridgeConfig
from new_mobile_ov.smolvlm2 import SmolVLMModel, load_smolvlm2_from_ckpt


DreamLiteMode = Literal["generate", "edit"]

DREAMLITE_GENERATION_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def dreamlite_generation_texts(prompts: Sequence[str]) -> list[str]:
    prompts = _clean_prompts(prompts, "image")
    return [DREAMLITE_GENERATION_TEMPLATE.format(f"[Generate]: {prompt}") for prompt in prompts]


def dreamlite_generation_lengths(
    tokenizer,
    prompts: Sequence[str],
    *,
    prefix_tokens: int,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        text=dreamlite_generation_texts(prompts),
        padding=True,
        return_tensors="pt",
    )
    lengths = encoded.attention_mask.sum(dim=1) - int(prefix_tokens)
    return lengths.clamp(min=1, max=int(max_length))


class DreamLiteCondition(NamedTuple):
    """Condition contract consumed by DreamLite's internal text projection."""

    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor


def _clean_prompts(prompts: Sequence[str], fallback: str) -> list[str]:
    return [" ".join(str(value).strip().split()) or fallback for value in prompts]


class SmolVLM2MultimodalFeatureProvider(nn.Module):
    """Frozen SmolVLM2 forward for DreamLite generation and editing."""

    def __init__(
        self,
        bridge_cfg: BridgeConfig,
        dreamlite_cfg: DreamLiteBridgeConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.max_length = int(bridge_cfg.max_length)
        self.tokenizer_model_id = str(bridge_cfg.tokenizer_model_id)
        self.processor_model_id = str(dreamlite_cfg.processor_model_id)
        self.disable_image_splitting = bool(dreamlite_cfg.disable_image_splitting)
        checkpoint = ensure_smolvlm2_checkpoint(bridge_cfg.smolvlm2_ckpt_path)
        self.smolvlm2_model = load_smolvlm2_from_ckpt(
            checkpoint,
            device=device,
            model_class=SmolVLMModel,
        )
        self.smolvlm2_model.eval().requires_grad_(False)
        self.hidden_size = _detect_hidden_size(self.smolvlm2_model)
        self._cached_tokenizer = None
        self._cached_processor = None
        self.to(device=device, dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.smolvlm2_model.eval()
        return self

    def _tokenizer(self):
        tokenizer = self.smolvlm2_model.get_tokenizer()
        if tokenizer is None:
            if self._cached_tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    self._cached_tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_model_id,
                        trust_remote_code=True,
                    )
                except Exception:
                    config = getattr(getattr(self.smolvlm2_model, "_model", None), "config", None)
                    vocab_size = int(getattr(config, "vocab_size", 32000))
                    self._cached_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
            tokenizer = self._cached_tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        return tokenizer

    def _processor(self):
        if self._cached_processor is None:
            from transformers import AutoProcessor

            self._cached_processor = AutoProcessor.from_pretrained(
                self.processor_model_id,
                trust_remote_code=True,
            )
        return self._cached_processor

    @staticmethod
    def _hidden_layers(outputs) -> list[torch.Tensor]:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (list, tuple)):
            raise RuntimeError("SmolVLM2 did not return hidden states.")
        layers = [value for value in hidden_states if isinstance(value, torch.Tensor) and value.dim() == 3]
        if not layers:
            raise RuntimeError("SmolVLM2 returned no token hidden-state tensors.")
        return layers

    def _generate_features(self, prompts: Sequence[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        tokenizer = self._tokenizer()
        fallback = tokenizer.eos_token or tokenizer.pad_token or "image"
        prompts = _clean_prompts(prompts, fallback)
        texts = [f"[Generate]: {prompt}" for prompt in prompts]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self.smolvlm2_model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self.smolvlm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return self._hidden_layers(outputs), attention_mask

    def _edit_features(
        self,
        prompts: Sequence[str],
        images: Sequence[Image.Image],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if len(prompts) != len(images):
            raise ValueError(f"Expected one source image per prompt, got {len(images)} for {len(prompts)}.")
        processor = self._processor()
        prompts = _clean_prompts(prompts, "preserve the source image")
        texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "[Edit]: Preserve the source image where appropriate and apply this "
                                f"instruction: {prompt}"
                            ),
                        },
                    ],
                }
            ]
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
        encoded = processor(
            text=texts,
            images=[[image.convert("RGB")] for image in images],
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_image_splitting": not self.disable_image_splitting},
        )
        device = next(self.smolvlm2_model.parameters()).device
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"}
        }
        with torch.no_grad():
            outputs = self.smolvlm2_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        return self._hidden_layers(outputs), model_inputs["attention_mask"]

    def forward(
        self,
        prompts: Sequence[str],
        *,
        mode: DreamLiteMode,
        images: Sequence[Image.Image] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if mode == "generate":
            if images is not None:
                raise ValueError("Generation mode does not consume source images.")
            return self._generate_features(prompts)
        if mode == "edit":
            if images is None:
                raise ValueError("Edit mode requires source images.")
            return self._edit_features(prompts, images)
        raise ValueError(f"Unsupported DreamLite bridge mode: {mode}")


class MobileOVDreamLiteImageBridge(nn.Module):
    """Replace DreamLite's Qwen3-VL condition with one SmolVLM2 bridge."""

    def __init__(
        self,
        bridge_cfg: BridgeConfig,
        dreamlite_cfg: DreamLiteBridgeConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge_cfg = bridge_cfg
        self.dreamlite_cfg = dreamlite_cfg
        self.sequence_length = int(dreamlite_cfg.sequence_length)
        self.condition_dim = int(dreamlite_cfg.condition_dim)
        self.variable_length_generation = bool(dreamlite_cfg.variable_length_generation)
        self.native_tokenizer_path = str(dreamlite_cfg.native_tokenizer_path)
        self.native_generation_prefix_tokens = int(dreamlite_cfg.native_generation_prefix_tokens)
        self._native_tokenizer = None
        self.feature_provider = SmolVLM2MultimodalFeatureProvider(
            bridge_cfg,
            dreamlite_cfg,
            device=device,
            dtype=dtype,
        )
        source_dim = int(self.feature_provider.hidden_size)
        attention_dim = int(dreamlite_cfg.attention_dim)
        self.num_fuse_layers = int(dreamlite_cfg.num_fuse_layers)
        if self.num_fuse_layers < 1:
            raise ValueError("dreamlite_bridge.num_fuse_layers must be positive.")

        self.layer_weights = nn.Parameter(torch.zeros(self.num_fuse_layers))
        self.semantic_norm = nn.LayerNorm(source_dim)
        self.semantic_projection = nn.Linear(source_dim, attention_dim, bias=False)
        self.lexical_norm = nn.LayerNorm(source_dim)
        self.lexical_projection = nn.Linear(source_dim, attention_dim, bias=False)
        gate = min(max(float(dreamlite_cfg.lexical_gate_init), 1e-4), 1.0 - 1e-4)
        self.lexical_gate_logit = nn.Parameter(
            torch.tensor([math.log(gate / (1.0 - gate))], dtype=torch.float32)
        )
        self.source_output_norm = nn.LayerNorm(attention_dim)
        self.mode_embedding = nn.Embedding(2, attention_dim)
        self.queries = nn.Parameter(
            torch.randn(1, self.sequence_length, attention_dim, dtype=torch.float32) * 0.02
        )
        self.query_blocks = nn.ModuleList(
            [
                ImageQueryBlock(
                    attention_dim,
                    num_heads=int(dreamlite_cfg.num_heads),
                    ff_mult=int(dreamlite_cfg.ff_mult),
                    dropout=float(dreamlite_cfg.dropout),
                )
                for _ in range(int(dreamlite_cfg.num_layers))
            ]
        )
        self.condition_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, self.condition_dim),
        )
        self.to(device=device, dtype=dtype)

    def _get_native_tokenizer(self):
        if self._native_tokenizer is None:
            from transformers import Qwen2TokenizerFast

            self._native_tokenizer = Qwen2TokenizerFast.from_pretrained(
                self.native_tokenizer_path,
                local_files_only=True,
            )
        return self._native_tokenizer

    def _generation_query_mask(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        tokenizer = self._get_native_tokenizer()
        lengths = dreamlite_generation_lengths(
            tokenizer,
            prompts,
            prefix_tokens=self.native_generation_prefix_tokens,
            max_length=self.sequence_length,
        ).to(device=device)
        batch_length = int(lengths.max().item())
        positions = torch.arange(batch_length, device=device)
        return positions.unsqueeze(0) < lengths.unsqueeze(1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_provider.smolvlm2_model.eval()
        return self

    def promote_trainable_parameters_to_fp32(self) -> None:
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        frozen_prefix = "feature_provider.smolvlm2_model."
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith(frozen_prefix)
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("feature_provider.smolvlm2_model.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Invalid DreamLite bridge state: missing={missing}, unexpected={unexpected}"
            )

    def _fuse_source(self, hidden_layers: list[torch.Tensor]) -> torch.Tensor:
        if len(hidden_layers) < self.num_fuse_layers:
            raise RuntimeError(
                f"Need {self.num_fuse_layers} SmolVLM2 layers, got {len(hidden_layers)}."
            )
        weights = torch.softmax(self.layer_weights, dim=0)
        semantic = torch.zeros_like(hidden_layers[-1])
        for weight, hidden in zip(weights, hidden_layers[-self.num_fuse_layers :]):
            semantic = semantic + weight.to(dtype=hidden.dtype) * hidden
        semantic = self.semantic_projection(self.semantic_norm(semantic))
        lexical = self.lexical_projection(self.lexical_norm(hidden_layers[0]))
        lexical_gate = torch.sigmoid(self.lexical_gate_logit).to(dtype=semantic.dtype)
        return self.source_output_norm(semantic + lexical_gate * lexical)

    def forward(
        self,
        prompts: List[str],
        *,
        mode: DreamLiteMode = "generate",
        images: Sequence[Image.Image] | None = None,
    ) -> DreamLiteCondition:
        hidden_layers, source_mask = self.feature_provider(
            prompts,
            mode=mode,
            images=images,
        )
        source = self._fuse_source(hidden_layers)
        mode_index = 0 if mode == "generate" else 1
        mode_value = self.mode_embedding.weight[mode_index].to(dtype=source.dtype)
        source = source + mode_value.view(1, 1, -1)
        if mode == "generate" and self.variable_length_generation:
            query_mask = self._generation_query_mask(prompts, device=source.device)
        else:
            query_mask = torch.ones(
                source.shape[0],
                self.sequence_length,
                dtype=torch.bool,
                device=source.device,
            )
        query_length = query_mask.shape[1]
        queries = self.queries[:, :query_length].expand(source.shape[0], -1, -1).to(dtype=source.dtype)
        queries = queries + mode_value.view(1, 1, -1)
        for block in self.query_blocks:
            queries = block(queries, source, source_mask, query_mask=query_mask)
        prompt_embeds = self.condition_head(queries)
        prompt_embeds = prompt_embeds * query_mask.to(dtype=prompt_embeds.dtype).unsqueeze(-1)
        attention_mask = query_mask.to(dtype=torch.long)
        return DreamLiteCondition(prompt_embeds=prompt_embeds, attention_mask=attention_mask)
