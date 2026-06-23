# SmolVLM2-500M Integration (Experiment 1)

Module này cung cấp implementation thuần PyTorch của SmolVLM2-500M để thay thế understanding module của Omni-Video.

## Cách sử dụng

### Bước 1: Convert weight từ HuggingFace (chạy 1 lần)

```bash
# Cần conda env có transformers
conda activate smolvlm2
python tools/convert_weights/convert_smolvlm2_weight.py \
    --model-id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --output-path checkpoints/smolvlm2_500m/smolvlm2_500m.pt
```

### Bước 2: Extract features (không cần transformers)

```bash
# Dùng conda env omnivideo (không cần transformers)
conda activate omnivideo
python tools/data_prepare/smolvlm2_feature_extract.py \
    --data-file path/to/base_feature_list.txt \
    --result-folder path/to/smolvlm2_feats \
    --ckpt-path checkpoints/smolvlm2_500m/smolvlm2_500m.pt
```

### Bước 3: Training Omni-Video với features mới

Sử dụng `--result-folder` từ bước 2 làm input cho training pipeline Omni-Video.

## Kiến trúc

- `modeling_smolvlm2.py`: Wrapper class để load và forward pass
- `config_smolvlm2.py`: Configuration classes
- `load_smolvlm2.py`: Utility functions để load checkpoint

## Lưu ý

- Checkpoint đã convert chứa toàn bộ model object, không chỉ state_dict
- Không cần transformers library sau khi đã convert
- Model được serialize với tokenizer để có thể tokenize text

