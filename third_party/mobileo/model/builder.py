from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
import warnings
from mobileo.model import mobileoForInferenceLM
from mobileo.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(model_path):
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = mobileoForInferenceLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto"
    )
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


def load_pretrained_model_lmms_eval(model_path, **kwargs):
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = mobileoForInferenceLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
