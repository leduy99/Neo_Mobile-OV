"""
SmolVLM2VisionHead: VisionHead-style resampler for SmolVLM2
Clone from OmniVideo's VisionHead, adapted for SmolVLM2 (llm_hidden_size=1024)

This module converts variable-length SmolVLM2 hidden states [B, L, 1024]
to fixed query tokens [B, Q, 1152] that the DM_Adapter expects.
"""

import os
import torch
import torch.nn as nn

# Use same precision as OmniVideo
TRAINABLE_PRECISION = torch.float32


class SmolVLM2VisionHead(nn.Module):
    """
    VisionHead-style head for SmolVLM2:
    [B, L, 1024] -> [B, Q, 1152]
    
    Architecture copied from OmniVideo's VisionHead, only changed llm_hidden_size=1024.
    This ensures interface match with DM_Adapter (which was pretrained with VisionHead outputs).
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 1024,      # SmolVLM2 hidden size
        hidden_size: int = 1152,          # Adapter in_channels
        learnable_query_length: int = 1,  # Q=1 for T2V bring-up (can increase to 4 later)
        TRAINABLE_PRECISION: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.learnable_query_length = learnable_query_length

        # Learnable decoder queries (same as OmniVideo VisionHead)
        self.decoder_query = nn.Parameter(
            torch.randn((1, self.learnable_query_length, self.hidden_size), dtype=TRAINABLE_PRECISION),
            requires_grad=True,
        )

        # Transformer encoder-decoder (same as OmniVideo VisionHead)
        self.visionHeadAdapter = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=self.hidden_size,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.0,
            dtype=TRAINABLE_PRECISION,
        )

        # FC layer: map LLM hidden -> adapter input size
        self.fc = nn.Sequential(
            nn.Linear(llm_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(TRAINABLE_PRECISION)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (same as OmniVideo VisionHead)"""
        nn.init.normal_(self.decoder_query, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, smol_hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert SmolVLM2 hidden states to query tokens.
        
        Args:
            smol_hidden: [B, L, 1024] - SmolVLM2 hidden states (variable length L)
        
        Returns:
            vision_tokens: [B, Q, 1152] - Fixed query tokens for adapter
        """
        # Map LLM hidden -> adapter input size
        input_embeds = self.fc(smol_hidden)  # [B, L, 1152]
        
        # Transformer encoder-decoder: compress variable-length sequence to fixed queries
        vision_tokens = self.visionHeadAdapter(
            src=input_embeds,
            tgt=self.decoder_query.repeat(input_embeds.shape[0], 1, 1),  # [B, Q, 1152]
        )  # [B, Q, 1152]
        
        return vision_tokens

    def save_pretrained(self, output_dir: str, state_dict=None):
        """
        Save the model's state dictionary to a directory.
        
        Args:
            output_dir: Directory to save the model files.
            state_dict: State dictionary to save. If None, uses model's state_dict().
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    def load_checkpoint(self, checkpoint_dir: str, map_location=None):
        """
        Load weights from checkpoint.
        
        Args:
            checkpoint_dir: Directory containing pytorch_model.bin
            map_location: torch.load map_location, defaults to CPU
        """
        if map_location is None:
            map_location = torch.device('cpu')
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location=map_location
        )
        self.load_state_dict(state_dict)
        return self
