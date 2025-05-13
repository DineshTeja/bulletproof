"""Utilities for loading and setting up language models."""

import logging
import os
import random
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer, 
)
from huggingface_hub import login

from neuro240.utils.config import (
    DEVICE, 
    HF_TOKEN, 
    REASONING_TOKENS, 
    get_model_path,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_model(
    model_name: str, 
    device: Optional[torch.device] = None,
    login_to_hf: bool = True,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Set up a lightweight model with all necessary configurations.

    Args:
        model_name: Name of the model from LIGHTWEIGHT_MODELS
        device: Device to load the model on ("cuda", "cpu", etc.)
        login_to_hf: Whether to attempt login to Hugging Face

    Returns:
        tuple: (tokenizer, model)
    """
    if device is None:
        device = DEVICE
    
    # Get the actual model path
    model_path = get_model_path(model_name)
    logger.info(f"Loading {model_name} ({model_path}) on {device}...")

    # Login to Hugging Face if needed
    if login_to_hf and HF_TOKEN:
        login(token=HF_TOKEN)
        logger.info("Logged in to Hugging Face Hub")
    elif login_to_hf:
        logger.warning("No Hugging Face token found in environment. Skipping login.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Add reasoning tokens
    logger.info(f"Adding {len(REASONING_TOKENS)} reasoning tokens to tokenizer")
    tokenizer.add_tokens(REASONING_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        logger.info("No pad token found. Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Move model to the specified device
    model.to(device)

    logger.info(f"Successfully loaded {model_name}")
    return tokenizer, model


def prepare_prompt(question: str) -> str:
    """Prepare a prompt for structured reasoning.
    
    Args:
        question: The question to answer
        
    Returns:
        formatted prompt string
    """
    return (
        f"You should answer the following question in a structured analytical manner. Follow these rules:\n"
        f"1. Inside <think> </think> tags, break down the question into key concepts and reasoning steps.\n"
        f"2. Inside <verify> </verify> tags, verify the logic by checking calculations, consistency, or common mistakes.\n"
        f"3. Inside <conclude> </conclude> tags, ONLY state the final answer. Do not include explanations here.\n"
        f"\nQuestion: {question}\n"
    )


def save_model(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    output_dir: str
) -> None:
    """Save a fine-tuned model and tokenizer.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save to
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model to {output_dir}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved successfully")


def load_fine_tuned_model(
    model_dir: str,
    device: Optional[torch.device] = None
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a fine-tuned model from disk.
    
    Args:
        model_dir: Directory where the model is saved
        device: Device to load model on
        
    Returns:
        tuple: (tokenizer, model)
    """
    if device is None:
        device = DEVICE
        
    logger.info(f"Loading fine-tuned model from {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Move model to the specified device
    model.to(device)
    
    logger.info(f"Fine-tuned model loaded successfully")
    return tokenizer, model 