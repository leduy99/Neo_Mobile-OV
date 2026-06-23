"""
Configuration for SmolVLM2-500M model.

This is a simplified config that matches the structure needed
for loading converted checkpoints.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SmolVLMVisionConfig:
    """Vision encoder configuration"""
    hidden_size: int = 1152
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02


@dataclass
class SmolVLMTextConfig:
    """Text model (LLaMA) configuration"""
    vocab_size: int = 128256
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 128002
    tie_word_embeddings: bool = False


@dataclass
class SmolVLMConfig:
    """Main SmolVLM2 configuration"""
    vision_config: Optional[SmolVLMVisionConfig] = None
    text_config: Optional[SmolVLMTextConfig] = None
    image_token_id: int = 128257
    scale_factor: int = 2
    use_cache: bool = True
    pad_token_id: int = 128002
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = SmolVLMVisionConfig()
        if self.text_config is None:
            self.text_config = SmolVLMTextConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary (for loading from checkpoint)"""
        # Filter valid keys for nested configs
        vision_config = None
        if "vision_config" in config_dict:
            if isinstance(config_dict["vision_config"], dict):
                # Only use valid keys for SmolVLMVisionConfig
                valid_vision_keys = {
                    'hidden_size', 'intermediate_size', 'num_hidden_layers',
                    'num_attention_heads', 'num_channels', 'image_size',
                    'patch_size', 'hidden_act', 'layer_norm_eps',
                    'attention_dropout', 'initializer_range'
                }
                vision_dict = {
                    k: v for k, v in config_dict["vision_config"].items()
                    if k in valid_vision_keys
                }
                vision_config = SmolVLMVisionConfig(**vision_dict)
            else:
                vision_config = config_dict["vision_config"]
        
        text_config = None
        if "text_config" in config_dict:
            if isinstance(config_dict["text_config"], dict):
                # Only use valid keys for SmolVLMTextConfig
                valid_text_keys = {
                    'vocab_size', 'hidden_size', 'intermediate_size',
                    'num_hidden_layers', 'num_attention_heads',
                    'num_key_value_heads', 'max_position_embeddings',
                    'rms_norm_eps', 'pad_token_id', 'tie_word_embeddings'
                }
                text_dict = {
                    k: v for k, v in config_dict["text_config"].items()
                    if k in valid_text_keys
                }
                text_config = SmolVLMTextConfig(**text_dict)
            else:
                text_config = config_dict["text_config"]
        
        # Create main config - only use valid keys
        valid_main_keys = {
            'image_token_id', 'scale_factor', 'use_cache', 'pad_token_id'
        }
        main_config = {
            k: v for k, v in config_dict.items()
            if k in valid_main_keys
        }
        
        return cls(
            vision_config=vision_config,
            text_config=text_config,
            **main_config
        )
    
    def to_dict(self):
        """Convert config to dictionary (for saving checkpoint)"""
        return {
            "vision_config": self.vision_config.__dict__ if self.vision_config else None,
            "text_config": self.text_config.__dict__ if self.text_config else None,
            "image_token_id": self.image_token_id,
            "scale_factor": self.scale_factor,
            "use_cache": self.use_cache,
            "pad_token_id": self.pad_token_id,
        }


