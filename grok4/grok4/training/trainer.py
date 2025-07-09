"""
Grok4 Trainer

Advanced training loop with mixed precision, gradient accumulation,
and distributed training support.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
import math
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger
from ..models.grok4_model import Grok4Model
from .data_loader import TextDataset
from .optimizer import get_optimizer, get_scheduler

logger = get_logger(__name__)


class Trainer:
    """
    Advanced trainer for Grok4 models with support for:
    - Mixed precision training
    - Gradient accumulation
    - Distributed training
    - Checkpointing and resuming
    - Wandb logging
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: Grok4Model,
        config: Optional[Union[str, Path, Config]] = None,
        tokenizer=None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Grok4Model instance
            config: Training configuration
            tokenizer: Tokenizer for text processing
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name
        """
        self.model = model
        
        # Load configuration
        if config is None:
            self.config = Config()
        elif isinstance(config, (str, Path)):
            self.config = Config.from_file(config)
        else:
            self.config = config
        
        self.training_config = self.config.training
        self.tokenizer = tokenizer
        
        # Device and distributed setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.local_rank = 0
        self.world_size = 1
        
        if self.is_distributed:
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Initialize mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.training_config.use_mixed_precision else None
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Logging
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_initialized = False
        
        logger.info(f"Trainer initialized on device: {self.device}")
        if self.is_distributed:
            logger.info(f"Distributed training: rank {self.local_rank}/{self.world_size}")
    
    def setup_training(self, train_dataset, val_dataset=None):
        """Setup training components."""
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.dataloader_num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_loader = None
        
        # Setup optimizer
        self.optimizer = get_optimizer(
            self.model,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        
        # Setup learning rate scheduler
        total_steps = min(
            self.training_config.max_steps,
            len(self.train_loader) * 1000  # Assume max 1000 epochs
        )
        
        self.lr_scheduler = get_scheduler(
            self.optimizer,
            warmup_steps=self.training_config.warmup_steps,
            total_steps=total_steps,
        )
        
        # Enable gradient checkpointing if requested
        if self.training_config.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model.module, 'transformer'):
                self.model.module.transformer.enable_gradient_checkpointing()
        
        # Initialize wandb
        if self.wandb_project and self.local_rank == 0:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.config.to_dict(),
                resume="allow",
            )
            self.wandb_initialized = True
    
    def train(self, dataset_path: str, output_dir: str, val_dataset_path: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Directory to save checkpoints
            val_dataset_path: Path to validation dataset
        """
        logger.info("Starting training...")
        
        # Load datasets
        train_dataset = TextDataset(dataset_path, tokenizer=self.tokenizer, max_length=self.config.model.max_seq_len)
        val_dataset = None
        if val_dataset_path:
            val_dataset = TextDataset(val_dataset_path, tokenizer=self.tokenizer, max_length=self.config.model.max_seq_len)
        
        # Setup training
        self.setup_training(train_dataset, val_dataset)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        while self.global_step < self.training_config.max_steps:
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch}",
                disable=self.local_rank != 0
            )
            
            for batch in progress_bar:
                loss = self.training_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Update progress bar
                if self.local_rank == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                        'step': self.global_step
                    })
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    self.log_metrics({
                        'train/loss': loss,
                        'train/learning_rate': self.lr_scheduler.get_last_lr()[0],
                        'train/global_step': self.global_step,
                        'train/epoch': self.current_epoch,
                    })
                
                # Validation
                if (self.val_loader and 
                    self.global_step % self.training_config.eval_steps == 0 and 
                    self.global_step > 0):
                    val_loss = self.evaluate()
                    self.log_metrics({'val/loss': val_loss})
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(output_path / "best_model")
                    
                    self.model.train()
                
                # Save checkpoint
                if (self.global_step % self.training_config.save_steps == 0 and 
                    self.global_step > 0 and 
                    self.local_rank == 0):
                    self.save_checkpoint(output_path / f"checkpoint-{self.global_step}")
                
                # Check if training is complete
                if self.global_step >= self.training_config.max_steps:
                    break
            
            self.current_epoch += 1
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            
            logger.info(f"Epoch {self.current_epoch - 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        if self.local_rank == 0:
            self.save_checkpoint(output_path / "final_model")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.wandb_initialized:
            wandb.finish()
    
    def training_step(self, batch) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
            # Scale loss for gradient accumulation
            loss = loss / self.training_config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (self.global_step + 1) % self.training_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                self.optimizer.step()
            
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.training_config.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", disable=self.local_rank != 0):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
            total_loss += loss.item()
            total_steps += 1
        
        avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
        
        if self.is_distributed:
            # Average across all processes
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, save_path: Path):
        """Save model checkpoint."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict
        model_state_dict = self.model.state_dict()
        if self.is_distributed:
            model_state_dict = self.model.module.state_dict()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, save_path / "trainer_state.pt")
        
        # Save model separately
        if hasattr(self.model, 'module'):
            self.model.module.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path / "trainer_state.pt", map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler state
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb and console."""
        if self.wandb_initialized and self.local_rank == 0:
            wandb.log(metrics, step=self.global_step)
        
        # Console logging
        if self.local_rank == 0:
            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
            logger.info(f"Step {self.global_step} | {log_str}")