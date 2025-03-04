import random
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across all random number generators.

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hle_subset(num_samples: int = 50):
    """
    Load a subset of the HLE dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        A dataset object containing the loaded samples
    """
    try:
        dataset = load_dataset("cais/hle", split="test")
        if len(dataset) > num_samples:
            subset_indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(subset_indices)
        return dataset
    except Exception as e:
        print(f"Error loading HLE dataset: {e}")
        return [
            {
                "question": f"Question {i}: What is 2x + 3 = 7?",
                "answer": "x = 2",
                "answer_type": "exactMatch",
            }
            for i in range(num_samples)
        ]


def prepare_prompt(question: str) -> str:
    """
    Prepare a prompt with structured reasoning instructions.

    Args:
        question: The question to be answered

    Returns:
        A formatted prompt string
    """
    return (
        f"You should answer the following question in a structured analytical manner. Follow these rules:\n"
        f"1. Inside <think> </think> tags, break down the question into key concepts and reasoning steps.\n"
        f"2. Inside <verify> </verify> tags, verify the logic by checking calculations, consistency, or common mistakes.\n"
        f"3. Inside <conclude> </conclude> tags, ONLY state the final answer. Do not include explanations here.\n"
        f"\nQuestion: {question}\n"
    )


def compute_embedding_similarity(answer1: str, answer2: str) -> float:
    """
    Compute semantic similarity between two text strings using embeddings.

    Args:
        answer1: First text string
        answer2: Second text string

    Returns:
        Cosine similarity score between the two embeddings
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = embedding_model.encode(answer1, convert_to_tensor=True)
    emb2 = embedding_model.encode(answer2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()
