from dataclasses import dataclass, field
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from model import *
from peft import PeftModel
import argparse
from pathlib import Path
import re
import shutil
import json
import math
from safetensors.torch import save_file



@dataclass
class T2IConfig:
    base_model_path: str = "checkpoints/Mobile-O-Post-Train-0.5B"
    lora_checkpoint_path: str = ""
    dtype: torch.dtype = torch.bfloat16
    use_lora_checkpoint: bool = True


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    checkpoint_dirs = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", item.name)
            if match:
                checkpoint_dirs.append((int(match.group(1)), item))

    if not checkpoint_dirs:
        print(f"Warning: No checkpoint directories found in {checkpoint_dir}")
        return None

    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_dir = checkpoint_dirs[0]
    global_step_dir = latest_dir / f"global_step{latest_step}"

    if not global_step_dir.exists():
        return None

    print(f"Found latest checkpoint: {latest_dir.name} (step {latest_step})")
    return str(global_step_dir)


class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = "cuda:0"
        self._load_models()

    def _load_deepspeed_state_dict(self, checkpoint_path):
        """Load and return state dict from a DeepSpeed checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        model_state_path = checkpoint_path / "mp_rank_00_model_states.pt"

        if not model_state_path.exists():
            print(f"Warning: Model states not found at {model_state_path}")
            return None

        print(f"Loading model states from: {model_state_path}")
        checkpoint = torch.load(model_state_path, map_location="cpu")

        if "module" in checkpoint:
            state_dict = checkpoint["module"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        print(f"Loaded {len(state_dict)} parameters from DeepSpeed checkpoint")
        return state_dict

    def _load_models(self):
        """Load model with optional LoRA adapters and full checkpoint weights."""
        import warnings
        warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
        print(f"Loading base model from: {self.config.base_model_path}")

        base_model = mobileoForInferenceLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=self.config.dtype,
            device_map="cpu",
        )

        if self.config.use_lora_checkpoint and self.config.lora_checkpoint_path:
            print(f"Loading LoRA checkpoint from: {self.config.lora_checkpoint_path}")

            # Load full checkpoint weights (DiT, projectors, etc.)
            ds_state_dict = self._load_deepspeed_state_dict(
                find_latest_checkpoint(self.config.lora_checkpoint_path) or self.config.lora_checkpoint_path
            )
            if ds_state_dict:
                missing, unexpected = base_model.load_state_dict(ds_state_dict, strict=False, assign=True)
                
            # Apply LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model, self.config.lora_checkpoint_path, torch_dtype=self.config.dtype
            )
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora" in n.lower())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA parameters: {lora_params:,} ({100 * lora_params / total_params:.2f}%)")
        else:
            self.model = base_model

        self.model = self.model.to(self.device).eval()

        tokenizer_path = self.config.lora_checkpoint_path if self.config.use_lora_checkpoint else self.config.base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Model loading complete. Ready for inference.")

    def save_merged_model(self, output_path: str, deepspeed_checkpoint_path: str = None):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        deepspeed_state_dict = None
        if deepspeed_checkpoint_path:
            deepspeed_state_dict = self._load_deepspeed_state_dict(deepspeed_checkpoint_path)

        # Merge LoRA if applicable
        if isinstance(self.model, PeftModel):
            print("Merging LoRA weights into base model...")
            merged_model = self.model.merge_and_unload().cpu()
        else:
            merged_model = self.model.cpu()

        # Prepare state dict
        config = merged_model.config if hasattr(merged_model, "config") else AutoConfig.from_pretrained(
            self.config.base_model_path, trust_remote_code=True
        )
        state_dict = merged_model.state_dict()

        # Merge DeepSpeed weights if available
        if deepspeed_state_dict:
            cleaned = {k.replace("module.", ""): v for k, v in deepspeed_state_dict.items()}
            state_dict.update(cleaned)
            print(f"Merged {len(cleaned)} parameters from DeepSpeed")

        # Remove PEFT-related keys
        peft_keys = [k for k in state_dict if any(x in k for x in ["lora_", "adapter_", "peft_"])]
        for k in peft_keys:
            del state_dict[k]

        # Save config
        config.save_pretrained(str(output_path))

        # Save weights (shard if > 5GB)
        total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
        max_shard_size = 5 * 1024**3

        if total_size > max_shard_size:
            num_shards = math.ceil(total_size / max_shard_size)
            keys = list(state_dict.keys())
            shard_size = len(keys) // num_shards + 1
            weight_map = {}

            for i in range(num_shards):
                shard_keys = keys[i * shard_size: (i + 1) * shard_size]
                shard_dict = {k: state_dict[k] for k in shard_keys}
                shard_filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
                save_file(shard_dict, str(output_path / shard_filename))
                for k in shard_keys:
                    weight_map[k] = shard_filename
                print(f"Saved shard {i+1}/{num_shards}: {shard_filename}")

            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            with open(output_path / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)
        else:
            save_file(state_dict, str(output_path / "model.safetensors"))

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        tokenizer.save_pretrained(str(output_path))

        # Copy additional files from base model
        base_path = Path(self.config.base_model_path)
        for py_file in base_path.glob("*.py"):
            if any(x in py_file.name.lower() for x in ["modeling", "configuration", "processing", "image"]):
                shutil.copy2(py_file, output_path / py_file.name)

        for json_file in ["generation_config.json", "preprocessor_config.json"]:
            src = base_path / json_file
            if src.exists():
                shutil.copy2(src, output_path / json_file)

        # Copy projector files
        for bin_file in ["mm_projector.bin", "gen_projector.bin"]:
            for search_path in [base_path, base_path / "merged_model"]:
                src = search_path / bin_file
                if src.exists():
                    shutil.copy2(src, output_path / bin_file)
                    break

        print(f"Model saved to: {output_path} ({len(state_dict)} params, {total_size / 1024**3:.2f}GB)")


def main():
    parser = argparse.ArgumentParser(description="Merge Mobile-O LoRA model with base model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--base_weights", type=str, required=True, help="Path to the base model weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for merged model")

    args = parser.parse_args()

    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if latest_checkpoint is None:
        print("Error: Could not find any valid checkpoints")
        return

    step_match = re.search(r"global_step(\d+)", latest_checkpoint)
    step_num = step_match.group(1) if step_match else "unknown"
    output_dir = args.output_dir or f"{args.checkpoint_dir}/final_merged_model_{step_num}"

    config = T2IConfig()
    config.base_model_path = args.base_weights
    config.lora_checkpoint_path = args.checkpoint_dir

    inference = TextToImageInference(config)
    inference.save_merged_model(output_dir, deepspeed_checkpoint_path=latest_checkpoint)
    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
