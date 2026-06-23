"""
Pure PyTorch implementation of SmolVLM2-500M model.

This is a wrapper that can load converted checkpoints without requiring transformers.
The actual model architecture is preserved from the converted checkpoint.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from dataclasses import dataclass

from .config_smolvlm2 import SmolVLMConfig

logger = logging.getLogger(__name__)


def _ensure_transformers_gelutanh_compat() -> None:
    """Back-compat for checkpoints serialized with older transformers classes."""
    try:
        import transformers.activations as hf_acts

        if not hasattr(hf_acts, "GELUTanh"):
            class GELUTanh(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return F.gelu(x, approximate="tanh")

            hf_acts.GELUTanh = GELUTanh
        if not hasattr(hf_acts, "SiLUActivation"):
            class SiLUActivation(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return F.silu(x)

            hf_acts.SiLUActivation = SiLUActivation
    except Exception:
        # Best-effort compatibility shim; if transformers is unavailable we continue.
        pass


def _patch_hf_config_compat(cfg) -> None:
    """Populate common generation/runtime fields expected by newer transformers."""
    if cfg is None:
        return

    fallback_fields = {
        "output_attentions": getattr(cfg, "_output_attentions", False),
        "output_hidden_states": getattr(cfg, "_output_hidden_states", False),
        "return_dict": getattr(cfg, "_return_dict", True),
        "use_return_dict": True,
        "use_cache": getattr(cfg, "use_cache", True),
    }
    for key, default_val in fallback_fields.items():
        if not hasattr(cfg, key):
            try:
                setattr(cfg, key, default_val)
            except Exception:
                pass


@dataclass
class SmolVLMOutput:
    """Output from SmolVLM2 forward pass"""
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class SmolVLMModel(nn.Module):
    """
    Wrapper for SmolVLM2 model that can load from converted checkpoints.
    
    This class loads a pre-serialized model object from a checkpoint file.
    The checkpoint should be created using the convert_weight script.
    """
    
    def __init__(self, config: Optional[SmolVLMConfig] = None):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None
        
    def load_from_checkpoint(self, ckpt_path: str, device: Optional[torch.device] = None):
        """
        Load model from converted checkpoint.
        
        Supports two formats:
        1. state_dict format (Option B1): Safe, no transformers needed at runtime
        2. Full object format (fallback): Requires transformers at runtime
        
        Args:
            ckpt_path: Path to the converted checkpoint file (.pt or .pth)
            device: Device to load the model on
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading SmolVLM2 model from {ckpt_path}")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        _ensure_transformers_gelutanh_compat()
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Prefer loading pre-serialized full model object when available.
        # This avoids fragile architecture reconstruction with mismatched
        # transformers internals (e.g. auto_docstring signature changes).
        if isinstance(checkpoint, dict) and checkpoint.get("model", None) is not None:
            self._model = checkpoint["model"]
            self._tokenizer = checkpoint.get("tokenizer", None)
            cfg_obj = checkpoint.get("config", None)
            if cfg_obj is not None:
                self.config = cfg_obj
                _patch_hf_config_compat(self.config)
            if hasattr(self._model, "config"):
                _patch_hf_config_compat(self._model.config)
            if hasattr(self._model, "to"):
                self._model = self._model.to(device)
            if hasattr(self._model, "eval"):
                self._model.eval()
            logger.info("✓ Model loaded successfully from full object checkpoint")
            return
        
        # Try Option B1: Load from state_dict (preferred, no transformers needed)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            try:
                logger.info("Attempting to load from state_dict (no transformers needed)...")
                
                # Load config - ưu tiên load trực tiếp từ checkpoint (không cần internet)
                if "config" in checkpoint and checkpoint["config"] is not None:
                    # Load trực tiếp từ checkpoint object (không cần internet) ✅
                    self.config = checkpoint["config"]
                    _patch_hf_config_compat(self.config)
                    logger.info("✓ Config loaded directly from checkpoint (no internet needed)")
                elif "config_dict" in checkpoint:
                    config_dict = checkpoint["config_dict"]
                    if isinstance(config_dict, dict):
                        # Try to reconstruct from dict (no internet needed)
                        try:
                            from transformers.models.smolvlm import configuration_smolvlm
                            transformers_config = configuration_smolvlm.SmolVLMConfig.from_dict(config_dict)
                            self.config = transformers_config
                            _patch_hf_config_compat(self.config)
                            logger.info("✓ Config reconstructed from dict (no internet needed)")
                        except Exception as e:
                            logger.warning(f"Could not reconstruct config from dict: {e}")
                            # Fallback: try HuggingFace (requires internet)
                            model_id = checkpoint.get("model_id", None)
                            if model_id:
                                try:
                                    from transformers import AutoConfig
                                    transformers_config = AutoConfig.from_pretrained(
                                        model_id, trust_remote_code=True
                                    )
                                    self.config = transformers_config
                                    _patch_hf_config_compat(self.config)
                                    logger.info("✓ Config loaded from HuggingFace (internet required)")
                                except Exception as e2:
                                    logger.warning(f"Could not load config from HuggingFace: {e2}")
                                    # Final fallback: use local config
                                    from .config_smolvlm2 import SmolVLMConfig
                                    self.config = SmolVLMConfig.from_dict(config_dict)
                                    _patch_hf_config_compat(self.config)
                            else:
                                # Use local config
                                from .config_smolvlm2 import SmolVLMConfig
                                self.config = SmolVLMConfig.from_dict(config_dict)
                                _patch_hf_config_compat(self.config)
                    else:
                        self.config = config_dict
                        _patch_hf_config_compat(self.config)
                else:
                    raise ValueError("Neither 'config' nor 'config_dict' found in checkpoint")
                
                # Build model from architecture
                # Check if we should load SmolVLMForConditionalGeneration (with lm_head) or SmolVLMModel (base)
                # If checkpoint has lm_head in state_dict, use ForConditionalGeneration
                has_lm_head = any('lm_head' in k for k in checkpoint["state_dict"].keys())
                try:
                    if has_lm_head:
                        # Load ForConditionalGeneration (with lm_head for text generation)
                        from .architecture_smolvlm2 import SmolVLMForConditionalGeneration
                        logger.info("Loading SmolVLMForConditionalGeneration (with lm_head for text generation)...")
                        self._model = SmolVLMForConditionalGeneration(self.config)
                    else:
                        # Load base SmolVLMModel (encoder-only)
                        from .architecture_smolvlm2 import SmolVLMModel
                        logger.info("Loading SmolVLMModel (base encoder-only)...")
                        self._model = SmolVLMModel(self.config)
                except (ImportError, NotImplementedError, Exception) as e:
                    logger.warning(f"Could not load from architecture: {e}")
                    logger.info("Falling back to full object format...")
                    raise
                
                # Load state_dict
                missing_keys, unexpected_keys = self._model.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys when loading state_dict: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                
                # Load tokenizer from vocab if available
                if "tokenizer_vocab" in checkpoint and checkpoint["tokenizer_vocab"] is not None:
                    try:
                        # Reconstruct tokenizer from vocab
                        # This is simplified - may need more tokenizer config
                        from transformers import PreTrainedTokenizerFast
                        if "tokenizer_config" in checkpoint:
                            self._tokenizer = PreTrainedTokenizerFast(
                                vocab=checkpoint["tokenizer_vocab"],
                                **checkpoint.get("tokenizer_config", {})
                            )
                        else:
                            logger.warning("Tokenizer config not found, tokenizer may not work correctly")
                    except Exception as e:
                        logger.warning(f"Could not reconstruct tokenizer from vocab: {e}")
                
                # Move to device and set eval mode
                self._model = self._model.to(device)
                self._model.eval()
                
                logger.info("✓ Model loaded successfully from state_dict (no transformers needed)")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load from state_dict: {e}")
                logger.info("Falling back to full object format...")
        
        # Fallback: Load full object (requires transformers)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                # Format: {"model": model_object, "tokenizer": tokenizer_object, ...}
                self._model = checkpoint["model"]
                self._tokenizer = checkpoint.get("tokenizer", None)
                if hasattr(checkpoint.get("config"), "__dict__"):
                    self.config = checkpoint.get("config")
                    _patch_hf_config_compat(self.config)
            else:
                # Assume the whole dict is the model
                self._model = checkpoint
        else:
            # Direct model object
            self._model = checkpoint

        if hasattr(self._model, "config"):
            _patch_hf_config_compat(self._model.config)
        
        # Move to device and set eval mode
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        
        logger.info("✓ Model loaded successfully from full object (requires transformers)")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, SmolVLMOutput]:
        """
        Forward pass through SmolVLM2 model.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            inputs_embeds: Input embeddings [B, L, D] (alternative to input_ids)
            pixel_values: Image pixel values (optional, for vision)
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return as dict or tuple
            
        Returns:
            If return_dict=True: SmolVLMOutput with last_hidden_state
            If return_dict=False: tuple of (last_hidden_state, hidden_states, attentions)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_checkpoint() first.")
        
        # Prepare inputs
        model_inputs = {}
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        
        model_inputs["output_hidden_states"] = output_hidden_states
        model_inputs["return_dict"] = True
        model_inputs.update(kwargs)
        
        # Forward pass
        # Use no_grad() instead of inference_mode() to allow gradients to flow
        # when model is frozen but projection layer needs gradients
        # inference_mode() would block ALL gradients, even for downstream layers
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self._model(**model_inputs)
        
        # Extract hidden states
        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)):
            last_hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, (list, tuple)):
            last_hidden = outputs[0]
        else:
            raise ValueError(f"Unexpected output format: {type(outputs)}")
        
        if return_dict:
            return SmolVLMOutput(
                last_hidden_state=last_hidden,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )
        else:
            hidden_states = getattr(outputs, "hidden_states", None)
            attentions = getattr(outputs, "attentions", None)
            return (last_hidden, hidden_states, attentions)
    
    def get_tokenizer(self):
        """Get the tokenizer if available"""
        return self._tokenizer
    
    def encode_text(
        self,
        text: Union[str, list],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Encode text using the model's tokenizer.
        
        Args:
            text: Text string or list of strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ("pt", "np", etc.)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available. Model checkpoint may not include tokenizer.")
        
        return self._tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )


class SmolVLMForConditionalGeneration(SmolVLMModel):
    """
    SmolVLM2 model with language modeling head for generation.
    
    This is a wrapper that can load converted checkpoints.
    It ensures the underlying model is SmolVLMForConditionalGeneration from architecture.
    """
    
    def load_from_checkpoint(self, ckpt_path: str, device: Optional[torch.device] = None):
        """
        Load ForConditionalGeneration model from checkpoint.
        Override to ensure we load SmolVLMForConditionalGeneration from architecture.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading SmolVLM2 ForConditionalGeneration from {ckpt_path}")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        _ensure_transformers_gelutanh_compat()
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and checkpoint.get("model", None) is not None:
            self._model = checkpoint["model"]
            self._tokenizer = checkpoint.get("tokenizer", None)
            cfg_obj = checkpoint.get("config", None)
            if cfg_obj is not None:
                self.config = cfg_obj
                _patch_hf_config_compat(self.config)
            if hasattr(self._model, "config"):
                _patch_hf_config_compat(self._model.config)
            if hasattr(self._model, "to"):
                self._model = self._model.to(device)
            if hasattr(self._model, "eval"):
                self._model.eval()
            logger.info("✅ SmolVLMForConditionalGeneration loaded from full object")
            return
        
        # Try to load from state_dict with ForConditionalGeneration architecture
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            try:
                logger.info("Loading SmolVLMForConditionalGeneration from state_dict...")
                
                # Load config
                if "config" in checkpoint and checkpoint["config"] is not None:
                    self.config = checkpoint["config"]
                    _patch_hf_config_compat(self.config)
                elif "config_dict" in checkpoint:
                    config_dict = checkpoint["config_dict"]
                    if isinstance(config_dict, dict):
                        try:
                            from transformers.models.smolvlm import configuration_smolvlm
                            self.config = configuration_smolvlm.SmolVLMConfig.from_dict(config_dict)
                            _patch_hf_config_compat(self.config)
                        except Exception as e:
                            logger.warning(f"Could not reconstruct config: {e}")
                            from .config_smolvlm2 import SmolVLMConfig
                            self.config = SmolVLMConfig.from_dict(config_dict)
                            _patch_hf_config_compat(self.config)
                    else:
                        self.config = config_dict
                        _patch_hf_config_compat(self.config)
                else:
                    raise ValueError("Config not found in checkpoint")
                
                # Build ForConditionalGeneration model from architecture
                from .architecture_smolvlm2 import SmolVLMForConditionalGeneration as ArchForCondGen
                self._model = ArchForCondGen(self.config)
                
                # Load state_dict
                missing_keys, unexpected_keys = self._model.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
                
                # Load tokenizer
                if "tokenizer_vocab" in checkpoint and checkpoint["tokenizer_vocab"] is not None:
                    try:
                        from transformers import PreTrainedTokenizerFast
                        if "tokenizer_config" in checkpoint:
                            self._tokenizer = PreTrainedTokenizerFast(
                                vocab=checkpoint["tokenizer_vocab"],
                                **checkpoint.get("tokenizer_config", {})
                            )
                    except Exception as e:
                        logger.warning(f"Could not reconstruct tokenizer: {e}")
                
                # Move to device and set eval mode
                self._model = self._model.to(device)
                self._model.eval()
                
                logger.info("✅ SmolVLMForConditionalGeneration loaded successfully")
                if hasattr(self._model, 'lm_head'):
                    logger.info("  - ✅ Has lm_head (text generation enabled)")
                if hasattr(self._model, 'generate'):
                    logger.info("  - ✅ Has generate() method")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load from state_dict: {e}")
                logger.info("Falling back to parent load_from_checkpoint...")
        
        # Fallback to parent implementation
        super().load_from_checkpoint(ckpt_path, device)
    
    def generate(self, *args, **kwargs):
        """
        Generate text using the underlying model's generate method.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_checkpoint() first.")
        
        if not hasattr(self._model, 'generate'):
            raise RuntimeError("Model does not have generate() method. Use SmolVLMForConditionalGeneration.")
        
        return self._model.generate(*args, **kwargs)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with language modeling head.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_checkpoint() first.")
        
        # If model has forward method that handles lm_head, use it directly
        if hasattr(self._model, 'lm_head'):
            # Use the model's forward which includes lm_head
            return self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        else:
            # Fallback to parent forward
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
