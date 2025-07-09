"""
Grok4 Embeddings

Token and positional embeddings for the Grok4 model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class Grok4Embeddings(nn.Module):
    """
    Grok4 embeddings combining token embeddings with learned positional embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Position embeddings (learned)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.pad_token_id].fill_(0)
        
        # Position embeddings
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)
            
        Returns:
            Embedded representations [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Scale embeddings by sqrt(d_model) for better training stability
        embeddings = embeddings * math.sqrt(self.d_model)
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings