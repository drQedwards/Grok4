"""
Data Utilities

Utility functions for data preprocessing and dataset loading.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator
import random

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load dataset from various formats.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        List of data samples
    """
    file_path = Path(file_path)
    data = []
    
    if file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
    
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    
    elif file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split by double newlines for paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            data = [{'text': p} for p in paragraphs]
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def preprocess_text(text: str, clean_whitespace: bool = True, remove_urls: bool = True) -> str:
    """
    Preprocess text for training.
    
    Args:
        text: Input text
        clean_whitespace: Whether to clean whitespace
        remove_urls: Whether to remove URLs
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Clean whitespace
    if clean_whitespace:
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
    
    return text


def split_dataset(
    data: List[Dict], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data: List of data samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train:n_train + n_val]
    test_data = shuffled_data[n_train + n_val:]
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def save_dataset(data: List[Dict], file_path: Union[str, Path], format: str = 'jsonl'):
    """
    Save dataset to file.
    
    Args:
        data: List of data samples
        file_path: Output file path
        format: Output format ('jsonl', 'json')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'jsonl':
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    elif format == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {len(data)} samples to {file_path}")


def create_chat_dataset(conversations: List[List[Dict]], output_path: Union[str, Path]):
    """
    Create a chat dataset from conversations.
    
    Args:
        conversations: List of conversations (each conversation is a list of messages)
        output_path: Output file path
    """
    chat_data = []
    
    for conversation in conversations:
        if len(conversation) >= 2:  # At least one exchange
            chat_data.append({'messages': conversation})
    
    save_dataset(chat_data, output_path)
    logger.info(f"Created chat dataset with {len(chat_data)} conversations")


def create_text_dataset(texts: List[str], output_path: Union[str, Path]):
    """
    Create a text dataset from raw texts.
    
    Args:
        texts: List of text strings
        output_path: Output file path
    """
    text_data = []
    
    for text in texts:
        cleaned_text = preprocess_text(text)
        if cleaned_text:  # Only add non-empty texts
            text_data.append({'text': cleaned_text})
    
    save_dataset(text_data, output_path)
    logger.info(f"Created text dataset with {len(text_data)} samples")


def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 100, 
    min_chunk_size: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Find a good break point (end of sentence)
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + min_chunk_size, end - 200), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """
    Merge multiple datasets.
    
    Args:
        *datasets: Variable number of datasets to merge
        
    Returns:
        Merged dataset
    """
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    logger.info(f"Merged {len(datasets)} datasets into {len(merged)} samples")
    return merged


def filter_dataset(
    data: List[Dict], 
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    text_field: str = 'text'
) -> List[Dict]:
    """
    Filter dataset by text length.
    
    Args:
        data: Input dataset
        min_length: Minimum text length
        max_length: Maximum text length
        text_field: Field containing text
        
    Returns:
        Filtered dataset
    """
    filtered = []
    
    for item in data:
        if text_field in item:
            text_len = len(item[text_field])
            
            # Apply length filters
            if min_length is not None and text_len < min_length:
                continue
            if max_length is not None and text_len > max_length:
                continue
            
            filtered.append(item)
    
    logger.info(f"Filtered dataset: {len(data)} -> {len(filtered)} samples")
    return filtered


def sample_dataset(data: List[Dict], n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Sample a subset of the dataset.
    
    Args:
        data: Input dataset
        n_samples: Number of samples to select
        seed: Random seed
        
    Returns:
        Sampled dataset
    """
    if n_samples >= len(data):
        return data
    
    random.seed(seed)
    sampled = random.sample(data, n_samples)
    
    logger.info(f"Sampled {n_samples} from {len(data)} samples")
    return sampled


def dataset_statistics(data: List[Dict], text_field: str = 'text') -> Dict:
    """
    Calculate dataset statistics.
    
    Args:
        data: Input dataset
        text_field: Field containing text
        
    Returns:
        Dictionary of statistics
    """
    if not data:
        return {}
    
    text_lengths = []
    for item in data:
        if text_field in item:
            text_lengths.append(len(item[text_field]))
    
    if not text_lengths:
        return {}
    
    stats = {
        'num_samples': len(data),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'min_text_length': min(text_lengths),
        'max_text_length': max(text_lengths),
        'total_characters': sum(text_lengths),
    }
    
    # Calculate percentiles
    sorted_lengths = sorted(text_lengths)
    n = len(sorted_lengths)
    stats['median_text_length'] = sorted_lengths[n // 2]
    stats['p25_text_length'] = sorted_lengths[n // 4]
    stats['p75_text_length'] = sorted_lengths[3 * n // 4]
    
    return stats