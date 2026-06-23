import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class mobileoTrainer(Trainer):

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
            
    def create_optimizer(self):
        """
        Setup the optimizer with different learning rates for different components.
        - model.diffusion_connector: 2e-4
        - model.dit: 1e-5
        - model.layers: 2e-5
        - model.vision_tower: 2e-5
        - model.embed_tokens.weight: 2e-5
        - lm_head.weight: 2e-5
        - mm_projector: self.args.mm_projector_lr (if specified)
        - other parameters: default learning rate
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Identify different parameter groups
            projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
            dit_parameters = [name for name, _ in opt_model.named_parameters() if "model.dit." in name]
            diffusion_connector_parameters = [name for name, _ in opt_model.named_parameters() if
                                              "model.diffusion_connector." in name]
            layers_parameters = [name for name, _ in opt_model.named_parameters() if "model.layers." in name]
            vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if
                                       "model.vision_tower." in name]
            embed_tokens_parameters = [name for name, _ in opt_model.named_parameters() if
                                       "model.embed_tokens.weight" in name]
            lm_head_parameters = [name for name, _ in opt_model.named_parameters() if "lm_head.weight" in name]

            optimizer_grouped_parameters = []

            # 1. Diffusion connector parameters (LR: 2e-4) - WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in decay_parameters and
                        n in diffusion_connector_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-4,
            })

            # 2. Diffusion connector parameters (LR: 2e-4) - WITHOUT decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n not in decay_parameters and
                        n in diffusion_connector_parameters and
                        p.requires_grad
                )],
                "weight_decay": 0.0,
                "lr": 1e-4,
            })

            # 3. DIT parameters (LR: 1e-5) - WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in decay_parameters and
                        n in dit_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-5,
            })

            # 4. DIT parameters (LR: 1e-5) - WITHOUT decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n not in decay_parameters and
                        n in dit_parameters and
                        p.requires_grad
                )],
                "weight_decay": 0.0,
                "lr": 1e-5,
            })

            # 5. model.layers parameters (LR: 2e-5) - WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in decay_parameters and
                        n in layers_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-6,
            })

            # 6. model.layers parameters (LR: 2e-5) - WITHOUT decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n not in decay_parameters and
                        n in layers_parameters and
                        p.requires_grad
                )],
                "weight_decay": 0.0,
                "lr": 1e-6,
            })

            # 7. model.vision_tower parameters (LR: 2e-5) - WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in decay_parameters and
                        n in vision_tower_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-6,
            })

            # 8. model.vision_tower parameters (LR: 2e-5) - WITHOUT decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n not in decay_parameters and
                        n in vision_tower_parameters and
                        p.requires_grad
                )],
                "weight_decay": 0.0,
                "lr": 1e-6,
            })

            # 9. model.embed_tokens.weight (LR: 2e-5) - typically WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in embed_tokens_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-6,
            })

            # 10. lm_head.weight (LR: 2e-5) - typically WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in lm_head_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
                "lr": 1e-6,
            })

            # 11. MM Projector parameters (custom LR if specified)
            if self.args.mm_projector_lr is not None:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (
                                n in decay_parameters and
                                n in projector_parameters and
                                n not in diffusion_connector_parameters and
                                n not in dit_parameters and
                                n not in layers_parameters and
                                n not in vision_tower_parameters and
                                n not in embed_tokens_parameters and
                                n not in lm_head_parameters and
                                p.requires_grad
                        )],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (
                                n not in decay_parameters and
                                n in projector_parameters and
                                n not in diffusion_connector_parameters and
                                n not in dit_parameters and
                                n not in layers_parameters and
                                n not in vision_tower_parameters and
                                n not in embed_tokens_parameters and
                                n not in lm_head_parameters and
                                p.requires_grad
                        )],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ])

            # 12. All other parameters (default LR) - WITH decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n in decay_parameters and
                        n not in projector_parameters and
                        n not in diffusion_connector_parameters and
                        n not in dit_parameters and
                        n not in layers_parameters and
                        n not in vision_tower_parameters and
                        n not in embed_tokens_parameters and
                        n not in lm_head_parameters and
                        p.requires_grad
                )],
                "weight_decay": self.args.weight_decay,
            })

            # 13. All other parameters (default LR) - WITHOUT decay
            optimizer_grouped_parameters.append({
                "params": [p for n, p in opt_model.named_parameters() if (
                        n not in decay_parameters and
                        n not in projector_parameters and
                        n not in diffusion_connector_parameters and
                        n not in dit_parameters and
                        n not in layers_parameters and
                        n not in vision_tower_parameters and
                        n not in embed_tokens_parameters and
                        n not in lm_head_parameters and
                        p.requires_grad
                )],
                "weight_decay": 0.0,
            })

            # Remove empty groups
            optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group["params"]) > 0]

            # Print parameter group info
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                print("\n=== Optimizer Parameter Groups ===")
                component_names = [
                    "diffusion_connector (decay)",
                    "diffusion_connector (no decay)",
                    "dit (decay)",
                    "dit (no decay)",
                    "layers (decay)",
                    "layers (no decay)",
                    "vision_tower (decay)",
                    "vision_tower (no decay)",
                    "embed_tokens",
                    "lm_head",
                    "mm_projector (decay)" if self.args.mm_projector_lr else None,
                    "mm_projector (no decay)" if self.args.mm_projector_lr else None,
                    "other (decay)",
                    "other (no decay)",
                ]
                component_names = [name for name in component_names if name is not None]

                for i, group in enumerate(optimizer_grouped_parameters):
                    lr = group.get('lr', self.args.learning_rate)
                    wd = group.get('weight_decay', 0.0)
                    name = component_names[i] if i < len(component_names) else f"group_{i}"
                    print(f"Group {i} ({name}): LR={lr:.2e}, Weight Decay={wd}, Params={len(group['params'])}")

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def create_optimizer_original(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

