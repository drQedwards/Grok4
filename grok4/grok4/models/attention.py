"""
Grok4 Attention Mechanism

Advanced multi-head attention with rotary position embedding (RoPE),
key-value caching, and optimized implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos and sin for efficiency
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        """Update cos/sin cache for the given sequence length."""
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
        
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Grok4Attention(nn.Module):
    """
    Multi-head attention with rotary position embedding.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Rotary position embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len=max_seq_len
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Flash attention support
        self.flash_attn = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through attention layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            past_key_value: Past key-value states for caching
            use_cache: Whether to return key-value for caching
            
        Returns:
            Tuple of (attention_output, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embedding
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle key-value caching
        if past_key_value is not None:
            # Concatenate past key-values
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Compute attention
        if self.flash_attn and attention_mask is None:
            # Use Flash Attention when available and no custom mask
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention computation
            attn_output = self._compute_attention(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value
    
    def _compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute standard scaled dot-product attention.
        """
        batch_size, n_heads, seq_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, kv_seq_len), dtype=torch.bool, device=query_states.device),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output