from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoTokenizer

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


from mobileo.constants import IMAGE_TOKEN_INDEX

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class mobileoFastSFTConfig(Qwen2Config):
    model_type = "llava_qwen2"


class mobileoFastSFTModel(LlavaMetaModel, Qwen2Model):
    config_class = mobileoFastSFTConfig

    def __init__(self, config: Qwen2Config):
        super(mobileoFastSFTModel, self).__init__(config)

class mobileoFastSFTForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = mobileoFastSFTConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = mobileoFastSFTModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    def anneal_conditioning_temperature(self, epoch: int, total_epochs: int):
        """
        Anneal the temperature of the mobile conditioning projector.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
            
        Returns:
            dict: Metrics including temperature and layer weights
        """
        model = self.get_model()
        
        if not hasattr(model, 'diffusion_connector'):
            print("[WARNING] Model doesn't have diffusion_connector")
            return None
        
        if not hasattr(model.diffusion_connector, 'anneal_temperature'):
            print("[WARNING] diffusion_connector doesn't have anneal_temperature")
            return None
        
        # Anneal the temperature
        model.diffusion_connector.anneal_temperature(epoch, total_epochs)
        
        # Get and return metrics
        if hasattr(model.diffusion_connector, 'get_metrics'):
            return model.diffusion_connector.get_metrics()
        
        return None
    
    def get_conditioning_metrics(self):
        """Get current conditioning metrics for logging."""
        model = self.get_model()
        
        if hasattr(model, 'diffusion_connector'):
            if hasattr(model.diffusion_connector, 'get_metrics'):
                return model.diffusion_connector.get_metrics()
        
        return None
    
    def visual(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        image_features = self.get_model().get_vision_tower()(pixel_values)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def prepare_inputs_for_sft(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, und_images, grid_thw, i_s_pos, image_sizes=None
    ):
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        
        vae = self.get_model().get_sana_vae()
        vae_device = vae.device
        prompt_image_embeds = vae.encode(gen_images.to(vae_device)).latent if gen_images is not None else None
        prompt_image_embeds = prompt_image_embeds * vae.config.scaling_factor if prompt_image_embeds is not None else None
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        image_idx = (input_ids == IMAGE_TOKEN_INDEX)
        text_embeds = self.get_model().embed_tokens(input_ids)
        labels[image_idx] = -100
        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ids: Optional[list] = None,
        i_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_image: Optional[torch.FloatTensor] = None,
        und_image: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents
            ) = self.prepare_inputs_for_sft(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        total_loss = None
        img_hidden_states = outputs.hidden_states


        assert latents is not None, "Currently we only support image loss when latents is None"
        noise = torch.randn_like(latents, device=latents.device)
        weighting_scheme = "uniform"
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
        sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        
        diffusion_pred = self.get_model().dit(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=self.get_model().diffusion_connector(img_hidden_states),
            encoder_attention_mask=attention_mask,
        ).sample

       
        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        diff_loss = torch.mean(
            (weighting.float() * (diffusion_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        diff_loss = diff_loss.mean()
        total_loss = diff_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

AutoConfig.register("llava_qwen2", mobileoFastSFTConfig)
AutoModelForCausalLM.register(mobileoFastSFTConfig, mobileoFastSFTForCausalLM)