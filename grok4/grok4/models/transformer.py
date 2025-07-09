"""
Grok4 Transformer Architecture

Advanced transformer implementation with optimized attention mechanisms,
feed-forward networks, and layer normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .attention import Grok4Attention


class Grok4MLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation function.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: SiLU(gate) * up"""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class Grok4Layer(nn.Module):
    """
    Single Grok4 transformer layer with pre-norm architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Pre-norm architecture
        self.input_layernorm = nn.LayerNorm(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)
        
        # Attention mechanism
        self.self_attn = Grok4Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Feed-forward network
        self.mlp = Grok4MLP(d_model, d_ff, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            past_key_value: Past key-value for caching
            use_cache: Whether to return key-value for caching
            
        Returns:
            Tuple of (output_states, present_key_value)
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        hidden_states = residual + attn_output
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class Grok4Transformer(nn.Module):
    """
    Grok4 Transformer stack with multiple layers.
    """
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.n_layers = n_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            Grok4Layer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(n_layers)
        ])
        
        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        Forward pass through all transformer layers.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            past_key_values: Past key-values for all layers
            use_cache: Whether to return key-values for caching
            
        Returns:
            Tuple of (output_states, present_key_values)
        """
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    use_cache,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                present_key_values.append(layer_outputs[1])
        
        if use_cache:
            return hidden_states, present_key_values
        else:
            return hidden_states