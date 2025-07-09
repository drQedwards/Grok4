"""
Data Loading Utilities

Dataset classes and data loading utilities for training Grok4 models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import random

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextDataset(Dataset):
    """
    Dataset for text training data in JSONL format.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 8192,
        stride: int = 512,
        text_field: str = "text",
    ):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to JSONL data file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            text_field: Field name containing text in JSON
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.text_field = text_field
        
        # Load and process data
        self.examples = self._load_data()
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if self.text_field in data:
                        text = data[self.text_field]
                        if isinstance(text, str) and len(text.strip()) > 0:
                            examples.extend(self._tokenize_and_chunk(text))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {line_num + 1}")
                    continue
        
        return examples
    
    def _tokenize_and_chunk(self, text: str) -> List[Dict]:
        """Tokenize text and split into chunks."""
        # Tokenize full text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        chunks = []
        
        # Split into overlapping chunks
        for i in range(0, len(tokens), self.stride):
            chunk_tokens = tokens[i:i + self.max_length]
            
            # Skip chunks that are too short
            if len(chunk_tokens) < 50:
                continue
            
            # Create input_ids and labels (for causal LM, they're the same)
            chunks.append({
                'input_ids': chunk_tokens,
                'labels': chunk_tokens.copy()
            })
        
        return chunks
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class ChatDataset(Dataset):
    """
    Dataset for chat/conversation data.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 8192,
        system_message: Optional[str] = None,
    ):
        """
        Initialize chat dataset.
        
        Args:
            data_path: Path to JSONL data file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            system_message: Optional system message to prepend
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message
        
        # Load data
        self.conversations = self._load_conversations()
        
        logger.info(f"Loaded {len(self.conversations)} conversations from {data_path}")
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversations from JSONL file."""
        conversations = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if 'messages' in data:
                        conversations.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {line_num + 1}")
                    continue
        
        return conversations
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation messages into training text."""
        formatted_text = ""
        
        # Add system message if provided
        if self.system_message:
            formatted_text += f"System: {self.system_message}\n"
        
        # Add conversation messages
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_text += f"System: {content}\n"
            elif role == 'user':
                formatted_text += f"User: {content}\n"
            elif role == 'assistant':
                formatted_text += f"Assistant: {content}\n"
        
        return formatted_text
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        messages = conversation['messages']
        
        # Format conversation
        text = self._format_conversation(messages)
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': tokens,
            'labels': tokens.copy()
        }


def collate_fn(batch):
    """
    Collate function for batching sequences.
    """
    # Get max length in batch
    max_length = max(len(item['input_ids']) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']
        
        # Pad sequences
        padding_length = max_length - len(input_ids)
        
        # For input_ids and labels, pad with tokenizer.pad_token_id
        # For attention_mask, pad with 0
        padded_input_ids = input_ids + [0] * padding_length  # Assuming 0 is pad_token_id
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        padded_labels = labels + [-100] * padding_length  # -100 is ignored in loss calculation
        
        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(padded_labels)
    
    return {
        'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
        'labels': torch.tensor(batch_labels, dtype=torch.long)
    }


def create_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )