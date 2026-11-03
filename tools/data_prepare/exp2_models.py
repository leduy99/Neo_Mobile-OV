"""Reuse installed VBench scorers and the existing Qwen/SigLIP model loaders."""
from __future__ import annotations

from pathlib import Path
import json
import importlib.metadata
import sys

import numpy as np
import torch

from new_mobile_ov.training.exp2_data import caption_instruction, subset_frames, validate_caption


ASSETS = {
    "clip": "clip_model/ViT-L-14.pt",
    "aesthetic": "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
    "musiq": "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
    "raft": "raft_model/models/raft-things.pth",
}


def cached_snapshot_info(policy):
    from huggingface_hub import snapshot_download

    report = {}
    for name in ("qwen", "siglip"):
        path = Path(snapshot_download(policy[f"{name}_model"], revision=policy[f"{name}_revision"], local_files_only=True))
        index = path / "model.safetensors.index.json"
        files = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        for filename in files:
            if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
                raise FileNotFoundError(f"Complete model weights are not cached: {path / filename}")
        report[name] = dict(snapshot=path.name, weight_files=len(files))
    return report


class QualityModels:
    def __init__(self, cache, repo, device="cuda:0"):
        import clip
        from pyiqa.archs.musiq_arch import MUSIQ
        from easydict import EasyDict

        sys.path.insert(0, str(Path(repo).resolve()))
        from vbench.third_party.RAFT.core.raft import RAFT
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

        paths = {k: Path(cache) / v for k, v in ASSETS.items()}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Reuse the VBench cache; missing {path}")
        self.device, self.padder = device, InputPadder
        self.clip, self.clip_preprocess = clip.load(str(paths["clip"]), device=device)
        self.clip.eval()
        self.aesthetic = torch.nn.Linear(768, 1).to(device)
        self.aesthetic.load_state_dict(torch.load(paths["aesthetic"], map_location="cpu", weights_only=True))
        self.aesthetic.eval()
        self.musiq = MUSIQ(pretrained_model_path=str(paths["musiq"])).to(device).eval()
        self.raft = RAFT(EasyDict(small=False, mixed_precision=False, alternate_corr=False))
        state = torch.load(paths["raft"], map_location="cpu", weights_only=True)
        self.raft.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.raft.to(device).eval()
        from tools.data_prepare.download_alignment_images import sha256_file
        self.provenance = dict(weights={key: dict(sha256=sha256_file(path), bytes=path.stat().st_size)
                                            for key, path in paths.items()},
                               runtime={name: importlib.metadata.version(name) for name in ("torch", "pyiqa", "scenedetect")},
                               raft_code_sha256=sha256_file(Path(repo) / "vbench/third_party/RAFT/core/raft.py"))

    @torch.inference_mode()
    def score(self, frames, policy):
        picked = subset_frames(frames, policy["quality_frames"])
        batch = torch.stack([self.clip_preprocess(im) for im in picked]).to(self.device)
        features = torch.nn.functional.normalize(self.clip.encode_image(batch).float(), dim=-1)
        aesthetic = self.aesthetic(features).flatten().cpu().tolist()
        musiq = []
        for im in picked:
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float().to(self.device) / 255
            musiq.append(float(self.musiq(pixels).item()))
        motion = []
        if len(frames) > 1:
            # Adjacent 24-fps pairs, not sparse pairs that overstate velocity.
            starts = np.linspace(0, len(frames) - 2, policy["quality_frames"]).round().astype(int)
            for start in starts:
                pair = []
                for im in frames[start:start + 2]:
                    im = im.copy()
                    im.thumbnail((320, 320))
                    pair.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float().to(self.device))
                pad = self.padder(pair[0].shape)
                first, second = pad.pad(*pair)
                _, flow = self.raft(first, second, iters=12, test_mode=True)
                motion.append(float(pad.unpad(flow).norm(dim=1).mean().item()))
        return dict(musiq_mean=float(np.mean(musiq)), musiq_frames=musiq,
                    aesthetic_mean=float(np.mean(aesthetic)), aesthetic_frames=aesthetic,
                    motion_mean_pixels=float(np.mean(motion)) if motion else None,
                    motion_pairs=motion, motion_grid_fps=24, motion_analysis_long_side=320)


class CaptionModels:
    def __init__(self, policy):
        from tools.verify_image_bridge_grounding_qwen36 import QwenVisualVerifier
        from tools.score_image_bridge_grounding_siglip import SiglipPairScorer

        count = torch.cuda.device_count()
        if count < 1 or (count == 1 and torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3):
            raise RuntimeError("Use two GPUs, or one >=100-GiB GPU, for Qwen plus SigLIP")
        from huggingface_hub import snapshot_download
        qwen_path = Path(snapshot_download(policy["qwen_model"], revision=policy["qwen_revision"], local_files_only=True))
        siglip_path = Path(snapshot_download(policy["siglip_model"], revision=policy["siglip_revision"], local_files_only=True))
        # Balance multiple GPUs; a single H200 has room for both cached models.
        self.verifier = QwenVisualVerifier(
            model_id=str(qwen_path), model_revision=qwen_path.name,
            dtype=torch.bfloat16, device_map="balanced", max_new_tokens=512,
            max_image_edge=512)
        self.siglip = SiglipPairScorer(model_id=str(siglip_path),
                                     revision=siglip_path.name,
                                     device=torch.device("cuda:0"), dtype=torch.bfloat16)
        self.provenance = dict(qwen_model=policy["qwen_model"],
                               qwen_revision=qwen_path.name,
                               siglip_model=policy["siglip_model"],
                               siglip_revision=siglip_path.name,
                               runtime={name: importlib.metadata.version(name) for name in ("torch", "transformers")},
                               caption_schema=policy["caption_schema"], source_caption_provided=False,
                               generation_passes_per_attempt=1,
                               visual_input="ordered_cropped_images_with_explicit_24fps_timestamps")

    @torch.inference_mode()
    def caption(self, frames, policy):
        from tools.verify_image_bridge_grounding_qwen36 import extract_json

        video = len(frames) > 1
        indices = np.linspace(0, len(frames) - 1, min(policy["caption_frames"], len(frames))).round().astype(int)
        content = [{"type": "text", "text": caption_instruction(video=video)}]
        for i in indices:
            if video:
                content.append({"type": "text", "text": f"Time {i / 24:.3f} seconds:"})
            content.append({"type": "image", "image": frames[i]})
        v = self.verifier
        inputs = v.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        device = v.model.get_input_embeddings().weight.device
        inputs = {k: x.to(device) if torch.is_tensor(x) else x for k, x in inputs.items()}
        output = v.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        response = v.processor.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        annotation = validate_caption(extract_json(response), policy, video=video)
        # Score the short caption: the old scorer truncates long captions at its context limit.
        logits = []
        if annotation["accepted"]:
            picked = subset_frames(frames, policy["quality_frames"])
            _, logits = self.siglip.score(picked, [annotation["caption_short"]] * len(picked))
        annotation.update(siglip_logits=logits, siglip_logit_mean=float(np.mean(logits)) if logits else None,
                          qwen_raw_response=response, **self.provenance)
        threshold = policy["siglip_min_logit"]
        if threshold is not None and annotation["siglip_logit_mean"] is not None and annotation["siglip_logit_mean"] < threshold:
            annotation["accepted"] = False
            annotation["reason"] += "; below configured SigLIP ranking threshold"
        return annotation
