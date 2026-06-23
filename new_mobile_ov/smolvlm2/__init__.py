"""
SmolVLM2-500M Model (Pure PyTorch Implementation)

This module provides a pure PyTorch implementation of SmolVLM2-500M
that can load weights from converted checkpoints without requiring
the transformers library.

For Experiment 1: Replace understanding module with SmolVLM2-500M
"""

from .modeling_smolvlm2 import SmolVLMModel, SmolVLMForConditionalGeneration
from .load_smolvlm2 import load_smolvlm2_from_ckpt

__all__ = [
    "SmolVLMModel",
    "SmolVLMForConditionalGeneration", 
    "load_smolvlm2_from_ckpt",
]


