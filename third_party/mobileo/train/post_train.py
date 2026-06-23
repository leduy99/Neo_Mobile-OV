import os
import copy
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random

import torch
import glob
import transformers
from PIL import ImageFile
from datasets import load_dataset, concatenate_datasets, Image, Features, Value
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from mobileo.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from mobileo.train.mobileo_post_trainer import mobileoTrainer
from mobileo import conversation as conversation_lib
from mobileo.model import mobileoFastForCausalLM
from mobileo.mm_utils import tokenizer_image_token
import warnings
warnings.filterwarnings("ignore", message="Plan failed with a CuDNNError")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

ImageFile.LOAD_TRUNCATED_IMAGES = True

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    diffusion_name_or_path: Optional[str] = field(default="Efficient-Large-Model/Sana_1600M_512px_diffusers")
    is_train: bool = field(default=False)
    full_ft: bool = field(
        default=False,
        metadata={"help": "Whether to do full FT for training"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for training"}
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA. Common choices: q_proj, k_proj, v_proj, o_proj for attention, gate_proj, up_proj, down_proj for FFN"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )
    lora_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load LoRA adapter weights from"}
    )
    save_merged_lora_model: bool = field(
        default=True,
        metadata={"help": "Whether to save a merged version of the model with LoRA weights"}
    )
    vlm_num_layers: int = field(default=4, metadata={"help": "Number of VLM layers."})
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    journeyDB_folder: Optional[str] = field(default=None)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "pad"
    aspect_ratio_size_und: List[float] = field(default_factory=lambda: [512.0, 512.0], metadata={"help": "Resolution for aspect ratio."},)
    aspect_ratio_size_gen: List[float] = field(default_factory=lambda: [512.0, 512.0], metadata={"help": "Resolution for aspect ratio."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    mm_projector_lr: Optional[float] = 1e-5
    group_by_modality_length: bool = field(default=False)
    ddp_find_unused_parameters: bool =True
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr':1e-6})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens <= 0: return
    input_embeddings = model.get_input_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_embeddings_avg

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources
def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    ignore_index: int = -100,
) -> Dict:
    input_ids_list = []
    labels_list = []

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eol_id = 198

    for i, source in enumerate(sources):
        if not isinstance(source, list):
            print(f"[WARN] Skipping corrupted sample {i}: not a list")
            continue

        tokens = []
        labels = []
        human_to_user = {
            "human": "user",
            "gpt": "assistant",
        }

        for j, msg in enumerate(source):
            text = msg["value"]

            role = human_to_user.get(msg["from"], msg["from"])

            text = f'<|im_start|>' + role + '\n' + msg['value']
            if "<image>" in text:
                text = text.replace("<image>\n", "")
                text = "<image>" + text

            msg_ids = tokenizer_image_token(text, tokenizer)
            tokens.extend(msg_ids)
            if msg["from"] == "human":
                labels.extend([ignore_index] * len(msg_ids))
            else:
                labels.extend(msg_ids)

            if msg["from"] != "human":
                tokens.append(im_end_id); tokens.append(eol_id)
                labels.append(im_end_id); labels.append(eol_id)  # Make IM_END learnable

        input_ids_tensor = torch.tensor(tokens, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        assert len(input_ids_tensor) == len(labels_tensor), f"Length mismatch: input_ids={len(input_ids_tensor)}, labels={len(labels_tensor)}"
        input_ids_list.append(input_ids_tensor)
        labels_list.append(labels_tensor)

    input_ids = torch.stack(input_ids_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return dict(input_ids=input_ids, labels=labels)


def parse_qna(example):
    qna_text = example.get("qna")
    parsed_qna = []
    if qna_text:
        if isinstance(qna_text, bytes):
            qna_text = qna_text.decode("utf-8", errors="ignore")
        lines = qna_text.strip().split("\n")
        for line in lines:
            if "Q:" in line and "A:" in line:
                q, a = line.split("A:", 1)
                question = q.replace("Q:", "").strip()
                answer = a.strip()
                parsed_qna.append({"question": question, "answer": answer})
    example["qna"] = parsed_qna
    return example
class LazySupervisedMixDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        ###################################### Dual Image Training #######################################
        data_files = glob.glob(os.path.join(self.data_args.image_folder, "*.tar"))

        # Define features with both und_image and gen_image
        features = Features({
            "und_image.jpg": Image(),      # Understanding image (real)
            "gen_image.png": Image(),      # Generation image (synthetic)
            "gen_prompt.txt": Value("string"),
            "caption.txt": Value("string"),
            "category.txt": Value("string"),
            "qna.txt": Value("string")
        })

        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files,
            split="train",
            num_proc=1,
            features=features
        )

        # Rename columns to match expected names
        train_dataset = train_dataset.rename_column("und_image.jpg", "und_image")
        train_dataset = train_dataset.rename_column("gen_image.png", "gen_image")
        train_dataset = train_dataset.rename_column("gen_prompt.txt", "gen_prompt")
        train_dataset = train_dataset.rename_column("caption.txt", "caption")
        train_dataset = train_dataset.rename_column("category.txt", "category")
        train_dataset = train_dataset.rename_column("qna.txt", "qna")

        # Add extra columns
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['DUAL_IMAGE'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])

        # Parse QNA
        #train_dataset = train_dataset.map(parse_qna, num_proc=1, writer_batch_size=100)
        
        # Keep only necessary columns
        keep_columns = ["und_image", "gen_image", "gen_prompt", "caption", "category", "type", "image_path", "qna"]
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names if col not in keep_columns])

        print(f"Finished loading {len(train_dataset)} dual-image samples")

        list_data_dict.append(train_dataset)

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Total number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "und_image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "und_image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _safe_img_process_und(self, imgs):
        try:
            out = []
            for img in imgs:
                ori_h, ori_w = img.height, img.width
                closest_size, closest_ratio = self.data_args.aspect_ratio_size_und, 2.0

                closest_size = [int(x) for x in closest_size]
                if closest_size[0] / ori_h > closest_size[1] / ori_w:
                    resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
                else:
                    resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ])
                out.append(transform(img))
            return out
        except Exception as e:
            print(f"Corrupted image during processing: {e}")
            return None
    def _safe_img_process_gen(self, imgs):
        try:
            out = []
            for img in imgs:
                ori_h, ori_w = img.height, img.width
                closest_size, closest_ratio = self.data_args.aspect_ratio_size_gen, 1.0

                closest_size = [int(x) for x in closest_size]
                if closest_size[0] / ori_h > closest_size[1] / ori_w:
                    resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
                else:
                    resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ])
                out.append(transform(img))
            return out
        except Exception as e:
            print(f"Corrupted image during processing: {e}")
            return None
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            try:
                sources = self.list_data_dict[i]
                qna_text = sources.get("qna")
                parsed_qna = []
                if qna_text:
                    if isinstance(qna_text, bytes):
                        qna_text = qna_text.decode("utf-8", errors="ignore")
                    if isinstance(qna_text, str):
                        lines = qna_text.strip().split("\n")
                        for line in lines:
                            if "Q:" in line and "A:" in line:
                                q, a = line.split("A:", 1)
                                question = q.replace("Q:", "").strip()
                                answer = a.strip()
                                parsed_qna.append({"question": question, "answer": answer})
                    elif isinstance(qna_text, list):
                        parsed_qna = qna_text
                sources["qna"] = parsed_qna
                                
                if sources["category"] == "text_and_image_to_image":
                    gen_prompt_text = f"Please edit the provided image according to the following description: {sources['gen_prompt']}"
                else:
                    gen_prompt_text = f"Please generate image based on the following caption: {sources['gen_prompt']}"

                if "qna" in sources and len(sources["qna"]) > 0:
                    qna_sample = random.choice(sources["qna"])
                    question = qna_sample["question"]
                    answer = qna_sample["answer"]
                else:
                    question = "Describe the image."
                    answer = "The image is so beautiful."

                sources["conversations"] = [
                    {"from": "human", "value": gen_prompt_text+"<image>"},
                    {"from": "gpt", "value": ""},
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},  # understanding target
                ]

                category = sources.get("category", "unknown")
                gen_image_file = self.list_data_dict[i]["gen_image"]
                if not isinstance(gen_image_file, list):
                    gen_image_file = [gen_image_file]

                gen_images = []
                for img in gen_image_file:
                    img = img.convert("RGB")
                    gen_images.append(img)

                # Load UNDERSTANDING image (real) - for understanding loss
                und_image_file = self.list_data_dict[i]["und_image"]
                if not isinstance(und_image_file, list):
                    und_image_file = [und_image_file]

                und_images = []
                for img in und_image_file:
                    img = img.convert("RGB")
                    und_images.append(img)

                processed_gen_images = self._safe_img_process_gen(gen_images)
                processed_und_images = self._safe_img_process_und(und_images)

                if processed_gen_images is None or processed_und_images is None:
                    print("Corrupted image during transform, picking new sample.")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue
                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
                data_dict = preprocess_qwen_2(sources, self.tokenizer, has_image=True)

                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

                data_dict["gen_image"] = processed_gen_images[0]
                data_dict["und_image"] = processed_und_images[0]
                data_dict["category"] = category
                return data_dict
            except Exception as e:
                print(f"[WARN] Skipping corrupted sample {i}: {e}")
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        batch_gen_images = []
        batch_und_images = []
        batch_categories = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"])
            if "category" in instance:
                batch_categories.append(instance["category"])

        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images.unsqueeze(0) for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None



        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
            if all(x is not None and x.shape == batch_und_images[0].shape for x in batch_und_images):
                batch['und_image'] = torch.stack(batch_und_images)
            else:
                batch['und_image'] = batch_und_images
        else:
            batch["und_image"] = None

        batch["categories"] = batch_categories if len(batch_categories) > 0 else None
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = mobileoFastForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )


    model.config.use_cache = False

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Configure LoRA if enabled
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Prepare model for k-bit training if using quantization
        if training_args.bits in [4, 8]:
            model = prepare_model_for_kbit_training(model)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Keep vision tower frozen
        for name, param in model.named_parameters():
            if "vision_tower" in name:
                param.requires_grad = True

        # Optionally train mm_projector
        if model_args.tune_mm_mlp_adapter:
            for name, param in model.named_parameters():
                if "mm_projector" in name:
                    param.requires_grad = True
    else:

        if model_args.full_ft:
            for n, p in model.get_model().named_parameters():
                p.requires_grad = True
            for n, p in model.get_vision_tower().named_parameters():
                 p.requires_grad = True
            for n, p in model.get_model().embed_tokens.named_parameters():
                p.requires_grad = True
            for n, p in model.lm_head.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.get_model().named_parameters():
                p.requires_grad = False
            for n, p in model.get_vision_tower().named_parameters():
                p.requires_grad = False
            for n, p in model.get_model().embed_tokens.named_parameters():
                p.requires_grad = False
            for n, p in model.lm_head.named_parameters():
                p.requires_grad = False
            # Unfreeze SA Proj (QKV) only
            for n, p in model.get_model().named_parameters():
                if any(x in n for x in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]):
                    p.requires_grad = True
                    print(f"Unfroze: {n}")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    # Initialize vision modules
    if model_args.use_lora:
        model.base_model.model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        image_processor = model.base_model.model.get_model().get_vision_tower().image_processor
    else:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        image_processor = model.get_model().get_vision_tower().image_processor

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # Calculate total and trainable parameters
    if model_args.use_lora:
        total_params = sum(p.numel() for p in model.base_model.model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if model_args.use_lora:
        model.base_model.model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    model.config.pad_token_id = tokenizer.pad_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = mobileoTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    model.config.is_train = False # Disable training mode to enable the DPM-Solver scheduler during inference

    # Save model with LoRA support
    if model_args.use_lora:
        # Save LoRA adapters
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Optionally save merged model
        if model_args.save_merged_lora_model:
            rank0_print("Merging LoRA weights and saving full model...")
            model = model.merge_and_unload()
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "merged_model"),
                vision_tower=model_args.vision_tower,
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            vision_tower=model_args.vision_tower,
        )


def load_lora_model(model_args, training_args):
    """
    Load a model with LoRA adapters for inference or continued training.
    """
    from peft import PeftModel

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )
    base_model = mobileoFastForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_args.lora_adapter_path,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    return model


if __name__ == "__main__":
    train()
