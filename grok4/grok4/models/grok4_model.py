"""
Grok4 Model - Main Model Class

This module contains the main Grok4Model class that serves as the primary interface
for the Grok4 AI framework.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import yaml
import json

from ..utils.config import Config
from ..utils.logger import get_logger
from .transformer import Grok4Transformer
from .embeddings import Grok4Embeddings

logger = get_logger(__name__)


class Grok4Model(nn.Module):
    """
    Main Grok4 Model class implementing advanced transformer architecture
    for language understanding and generation.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Path, Dict]] = None,
        vocab_size: int = 50432,
        n_layers: int = 24,
        n_heads: int = 16,
        d_model: int = 2048,
        d_ff: int = 8192,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        load_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize Grok4 Model.
        
        Args:
            config: Configuration file path or dictionary
            vocab_size: Vocabulary size
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_model: Model dimension
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            load_pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained model
        """
        super().__init__()
        
        # Load configuration
        if config is not None:
            if isinstance(config, (str, Path)):
                with open(config, 'r') as f:
                    if str(config).endswith('.yaml') or str(config).endswith('.yml'):
                        config_dict = yaml.safe_load(f)
                    else:
                        config_dict = json.load(f)
                self.config = Config(config_dict)
            else:
                self.config = Config(config)
        else:
            self.config = Config({
                'model': {
                    'vocab_size': vocab_size,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'd_model': d_model,
                    'd_ff': d_ff,
                    'max_seq_len': max_seq_len,
                    'dropout': dropout,
                }
            })
        
        # Extract model parameters
        model_config = self.config.model
        self.vocab_size = model_config.vocab_size
        self.n_layers = model_config.n_layers
        self.n_heads = model_config.n_heads
        self.d_model = model_config.d_model
        self.d_ff = model_config.d_ff
        self.max_seq_len = model_config.max_seq_len
        self.dropout = model_config.dropout
        
        # Initialize components
        self.embeddings = Grok4Embeddings(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )
        
        self.transformer = Grok4Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len
        )
        
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Load pretrained weights if specified
        if load_pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)
            
        logger.info(f"Initialized Grok4Model with {self.num_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Dict]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Past key-value states for efficient generation
            return_dict: Whether to return a dictionary
            use_cache: Whether to return past key values
            
        Returns:
            Logits or dictionary with logits and optional past key values
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Transformer layers
        transformer_outputs = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        if use_cache:
            hidden_states, present_key_values = transformer_outputs
        else:
            hidden_states = transformer_outputs
            present_key_values = None
        
        # Final layer norm and language modeling head
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': present_key_values if use_cache else None,
                'hidden_states': hidden_states,
            }
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
        tokenizer=None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            stop_tokens: List of stop tokens
            tokenizer: Tokenizer to use
            
        Returns:
            Generated text
        """
        if tokenizer is None:
            # Use a default tokenizer if none provided
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(next(self.parameters()).device)
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = self.forward(
                    input_ids=generated_ids,
                    use_cache=True,
                    return_dict=True
                )
            else:
                outputs = self.forward(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sampling
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for stop tokens
            if stop_tokens:
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if any(stop_token in generated_text for stop_token in stop_tokens):
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        tokenizer=None,
    ) -> str:
        """
        Chat completion interface.
        
        Args:
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tokenizer: Tokenizer to use
            
        Returns:
            Assistant response
        """
        # Convert messages to prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=["User:", "System:"],
            tokenizer=tokenizer,
        )
        
        return response.strip()
    
    def embed(self, texts: List[str], tokenizer=None) -> torch.Tensor:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer to use
            
        Returns:
            Embeddings tensor
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                input_ids = tokenizer.encode(text, return_tensors="pt")
                input_ids = input_ids.to(next(self.parameters()).device)
                
                outputs = self.forward(input_ids, return_dict=True)
                hidden_states = outputs['hidden_states']
                
                # Mean pooling
                embedding = hidden_states.mean(dim=1)
                embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def save_pretrained(self, save_path: str):
        """Save model to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save configuration
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load pretrained model from disk."""
        load_path = Path(load_path)
        
        # Load model weights
        state_dict = torch.load(load_path / "pytorch_model.bin", map_location="cpu")
        self.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {load_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from pretrained checkpoint."""
        model_path = Path(model_path)
        
        # Load configuration
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(config=config, **kwargs)
        
        # Load weights
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model