"""Data loading and preprocessing utilities."""

import os
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from neuro240.utils.config import DEFAULT_DATA_DIR, DEFAULT_EVALUATION_CONFIG
from neuro240.utils.types import DatasetItem

logger = logging.getLogger(__name__)


def load_hle_dataset(
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "test"
) -> Dataset:
    """Load HLE dataset and optionally limit to a subset.
    
    Args:
        num_samples: Maximum number of samples to load (None for all)
        cache_dir: Directory to cache the dataset
        split: Dataset split to load ("train", "test", etc.)
        
    Returns:
        Dataset object
    """
    if cache_dir is None:
        cache_dir = os.path.join(DEFAULT_DATA_DIR, "hle")
        
    try:
        logger.info(f"Loading HLE dataset (split={split}, num_samples={num_samples})")
        dataset = load_dataset("cais/hle", split=split, cache_dir=cache_dir)
        
        # Limit samples if requested
        if num_samples is not None and len(dataset) > num_samples:
            subset_indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(subset_indices)
            
        logger.info(f"Loaded {len(dataset)} HLE questions")
        
        # Ensure the dataset has the required fields
        required_fields = ["question", "answer", "answer_type"]
        missing_fields = [f for f in required_fields if f not in dataset.column_names]
        
        if missing_fields:
            raise ValueError(f"Dataset missing required fields: {missing_fields}")
            
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading HLE dataset: {e}")
        logger.warning("Falling back to dummy dataset")
        
        # Create a dummy dataset if the real one can't be loaded
        num_samples = num_samples or 50
        dummy_data = {
            "question": [f"Question {i}: What is 2x + 3 = 7?" for i in range(num_samples)],
            "answer": ["x = 2" for _ in range(num_samples)],
            "answer_type": ["exactMatch" for _ in range(num_samples)]
        }
        return Dataset.from_dict(dummy_data)


def create_splits(
    dataset: Dataset, 
    train_size: float = 0.7, 
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: The dataset to split
        train_size: Proportion for training set (0-1)
        val_size: Proportion for validation set (0-1) 
        test_size: Proportion for test set (0-1)
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate proportions
    total = train_size + val_size + test_size
    if not (0.99 <= total <= 1.01):  # Allow for small floating point errors
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Get indices and shuffle
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create dataset splits
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    logger.info(f"Split dataset into {len(train_dataset)} train, "
                f"{len(val_dataset)} validation, and {len(test_dataset)} test samples.")
    
    return train_dataset, val_dataset, test_dataset


def prepare_batch_for_training(
    batch_indices: List[int],
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch of examples for training.
    
    Args:
        batch_indices: List of indices into the dataset
        dataset: The dataset to pull examples from
        tokenizer: Tokenizer for encoding inputs
        device: Device to place tensors on
        
    Returns:
        Dict with input_ids, attention_mask, etc.
    """
    questions = [dataset[i]["question"] for i in batch_indices]
    prompts = [prepare_prompt(q) for q in questions]
    
    # Encode all prompts
    encodings = tokenizer(
        prompts,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    batch = {k: v.to(device) for k, v in encodings.items()}
    
    # Add reference to original examples
    batch["example_indices"] = batch_indices
    
    return batch


def prepare_prompt(question: str) -> str:
    """Prepare a structured reasoning prompt for the model.
    
    Args:
        question: Question to answer
        
    Returns:
        Formatted prompt string
    """
    return (
        f"You should answer the following question in a structured analytical manner. Follow these rules:\n"
        f"1. Inside <think> </think> tags, break down the question into key concepts and reasoning steps.\n"
        f"2. Inside <verify> </verify> tags, verify the logic by checking calculations, consistency, or common mistakes.\n"
        f"3. Inside <conclude> </conclude> tags, ONLY state the final answer. Do not include explanations here.\n"
        f"\nQuestion: {question}\n"
    ) 