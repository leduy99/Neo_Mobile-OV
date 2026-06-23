"""
Utility functions for loading SmolVLM2 models from converted checkpoints.
"""

import os
import logging
import torch
from typing import Optional

from .modeling_smolvlm2 import SmolVLMModel, SmolVLMForConditionalGeneration
from .config_smolvlm2 import SmolVLMConfig

logger = logging.getLogger(__name__)


def load_smolvlm2_from_ckpt(
    ckpt_path: str,
    device: Optional[torch.device] = None,
    model_class: type = SmolVLMModel,
) -> SmolVLMModel:
    """
    Load SmolVLM2 model from converted checkpoint.
    
    Args:
        ckpt_path: Path to the converted checkpoint file
        device: Device to load the model on (default: cuda if available)
        model_class: Model class to instantiate (SmolVLMModel or SmolVLMForConditionalGeneration)
        
    Returns:
        Loaded SmolVLM2 model instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class()
    model.load_from_checkpoint(ckpt_path, device=device)
    
    return model


