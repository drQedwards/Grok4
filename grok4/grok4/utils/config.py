"""
Configuration Management

Utilities for handling configuration files and parameters.
"""

import yaml
import json
from typing import Any, Dict, Union, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    vocab_size: int = 50432
    n_layers: int = 24
    n_heads: int = 16
    d_model: int = 2048
    d_ff: int = 8192
    max_seq_len: int = 8192
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    do_sample: bool = True
    batch_size: int = 1


class Config:
    """
    Configuration manager for Grok4.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dictionary."""
        if config_dict is None:
            config_dict = {}
        
        # Initialize sub-configurations
        self.model = ModelConfig(**config_dict.get('model', {}))
        self.training = TrainingConfig(**config_dict.get('training', {}))
        self.inference = InferenceConfig(**config_dict.get('inference', {}))
        
        # Store raw config for custom parameters
        self._raw_config = config_dict
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def save(self, save_path: Union[str, Path]):
        """Save configuration to file."""
        save_path = Path(save_path)
        config_dict = self.to_dict()
        
        with open(save_path, 'w') as f:
            if save_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {save_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'inference': asdict(self.inference),
            **{k: v for k, v in self._raw_config.items() 
               if k not in ['model', 'training', 'inference']}
        }
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (ModelConfig, TrainingConfig, InferenceConfig)):
                    # Update nested config
                    for nested_key, nested_value in value.items():
                        setattr(getattr(self, key), nested_key, nested_value)
                else:
                    setattr(self, key, value)
            else:
                self._raw_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._raw_config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._raw_config[key] = value
    
    def __str__(self) -> str:
        """String representation."""
        return f"Config({yaml.dump(self.to_dict(), default_flow_style=False)})"
    
    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()