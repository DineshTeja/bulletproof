import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import login
from openai import OpenAI

from models import ExtractedResponse
from utils import set_seed, load_hle_subset
from evaluation import generate_text, compute_reward, evaluate_model

# Enable CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set Hugging Face authentication token
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. Some operations may fail.")

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
else:
    print(
        "Warning: OPENAI_API_KEY environment variable not set. OpenAI operations will fail."
    )
    openai_client = OpenAI(api_key="your-key-will-be-set-from-env-var")

openai_model = "gpt-4o-mini"

# Set seed for reproducibility
set_seed(42)

# Switch to a lightweight model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add reasoning tokens
reasoning_tokens = [
    "<think>",
    "</think>",
    "<verify>",
    "</verify>",
    "<conclude>",
    "</conclude>",
]
tokenizer.add_tokens(reasoning_tokens)
model.resize_token_embeddings(len(tokenizer))

# Set pad token (some models don't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.to(device)

# Load HLE dataset subset
hle_dataset = load_hle_subset(50)
print(f"Loaded {len(hle_dataset)} HLE questions")

# Final evaluation
print("\n--- Evaluating baseline model ---")
baseline_reward = evaluate_model(
    model, tokenizer, hle_dataset, openai_client, openai_model, device
)

# Train model (training logic remains the same)

# Final evaluation after fine-tuning
print("\n--- Evaluating fine-tuned model ---")
final_reward = evaluate_model(
    model, tokenizer, hle_dataset, openai_client, openai_model, device
)
print(f"Performance Improvement: {final_reward - baseline_reward:.4f}")
