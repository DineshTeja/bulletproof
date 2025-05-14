"""Configuration management for the Neuro240 project."""

import os
import torch
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Load environment variables from .env file
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}. Using default environment.")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# API keys and tokens
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Default OpenAI model
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Available lightweight models
LIGHTWEIGHT_MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi": "microsoft/phi-2",
    "stablelm": "stabilityai/stablelm-3b-4e1t",
    "flan_t5_small": "google/flan-t5-small",
    "gpt2": "gpt2",
    "opt": "facebook/opt-1.3b",
}

# Default model
DEFAULT_MODEL = "phi"

# Custom tokenizer tokens
REASONING_TOKENS = [
    "<think>", "</think>", 
    "<verify>", "</verify>", 
    "<conclude>", "</conclude>"
]

# Default paths
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "models"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "results"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"

# Create directories if they don't exist
DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MODEL_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Default training settings
DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epsilon": 0.2,  # PPO clip parameter
    "value_coef": 0.5,  # Value loss coefficient
    "kl_coef": 0.1,  # KL penalty coefficient
    "max_grad_norm": 1.0,  # Gradient clipping
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "seed": 42,
}

# Default evaluation settings
DEFAULT_EVALUATION_CONFIG: Dict[str, Any] = {
    "num_samples": 50,
    "max_generation_length": 150,
    "temperature": 0.7,
}

# Default reward function coefficients
DEFAULT_REWARD_COEFFICIENTS: Dict[str, float] = {
    "lambda_logic": 0.3,  # Weight for logical consistency
    "lambda_step": 0.3,  # Weight for stepwise correctness
    "lambda_halluc": 0.1,  # Weight for hallucination penalty
    "lambda_wrong": 0.3,  # Weight for wrong answer penalty
}

def get_model_path(model_name: str) -> str:
    """Get the path to a pre-trained model by name."""
    if model_name in LIGHTWEIGHT_MODELS:
        return LIGHTWEIGHT_MODELS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(LIGHTWEIGHT_MODELS.keys())}")
    
def get_output_model_path(model_name: str) -> Path:
    """Get the path to save a fine-tuned model."""
    return DEFAULT_MODEL_DIR / model_name

def get_API_keys() -> Dict[str, Optional[str]]:
    """Get API keys from environment variables."""
    return {
        "hf_token": HF_TOKEN,
        "openai_api_key": OPENAI_API_KEY,
    }

def validate_API_keys() -> bool:
    """Validate that required API keys are available."""
    keys = get_API_keys()
    missing_keys = [key for key, value in keys.items() if value is None]
    
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
        logger.error("Please set these in your .env file.")
        return False
    
    return True 