"""
Optimizer and Scheduler Utilities

Utilities for creating optimizers and learning rate schedulers.
"""

import torch
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Optional, Union
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ("adamw", "adam")
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        eps: Adam epsilon
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Separate parameters by weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created {optimizer_name} optimizer")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear_warmup",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs
):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ("linear_warmup", "cosine_warmup", "constant")
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum learning rate ratio for cosine schedule
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_name.lower() == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            **kwargs
        )
    elif scheduler_name.lower() == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            **kwargs
        )
    elif scheduler_name.lower() == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=optimizer.defaults['lr'] * min_lr_ratio,
            **kwargs
        )
    elif scheduler_name.lower() == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    logger.info(f"Created {scheduler_name} scheduler")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Total steps: {total_steps}")
    
    return scheduler


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom scheduler combining linear warmup with cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr_ratio * base_lr + 
                (base_lr - self.min_lr_ratio * base_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


def get_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.01,
    layer_decay: Optional[float] = None,
) -> list:
    """
    Create parameter groups with optional layer-wise learning rate decay.
    
    Args:
        model: Model to create groups for
        learning_rate: Base learning rate
        weight_decay: Weight decay coefficient
        layer_decay: Layer decay factor (e.g., 0.8)
        
    Returns:
        List of parameter groups
    """
    if layer_decay is None:
        # Standard parameter grouping
        return get_optimizer(model, lr=learning_rate, weight_decay=weight_decay).param_groups
    
    # Layer-wise decay
    param_groups = []
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    # Get number of layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        num_layers = len(model.transformer.layers)
    else:
        logger.warning("Could not determine number of layers for layer decay")
        num_layers = 24  # Default assumption
    
    # Create groups for each layer
    for i, layer in enumerate(model.transformer.layers):
        layer_lr = learning_rate * (layer_decay ** (num_layers - i - 1))
        
        # Parameters with weight decay
        param_groups.append({
            "params": [
                p for n, p in layer.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": layer_lr,
            "weight_decay": weight_decay,
        })
        
        # Parameters without weight decay
        param_groups.append({
            "params": [
                p for n, p in layer.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": layer_lr,
            "weight_decay": 0.0,
        })
    
    # Add embeddings and head parameters
    other_params_with_decay = []
    other_params_no_decay = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith('transformer.layers'):
            if any(nd in name for nd in no_decay):
                other_params_no_decay.append(param)
            else:
                other_params_with_decay.append(param)
    
    if other_params_with_decay:
        param_groups.append({
            "params": other_params_with_decay,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    
    if other_params_no_decay:
        param_groups.append({
            "params": other_params_no_decay,
            "lr": learning_rate,
            "weight_decay": 0.0,
        })
    
    logger.info(f"Created {len(param_groups)} parameter groups with layer decay {layer_decay}")
    
    return param_groups