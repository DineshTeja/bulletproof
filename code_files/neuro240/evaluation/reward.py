"""Reward calculation for reinforcement learning."""

import logging
from typing import Dict, Any, Optional

from neuro240.utils.config import DEFAULT_REWARD_COEFFICIENTS
from neuro240.evaluation.metrics import (
    extract_reasoning_steps,
    build_reasoning_graph,
    evaluate_logical_consistency,
)
from neuro240.evaluation.correctness import (
    evaluate_answer_correctness,
    evaluate_stepwise_correctness,
)
from neuro240.evaluation.hallucination import detect_hallucinations

logger = logging.getLogger(__name__)


def compute_reward(
    question: str,
    generated_output: Dict[str, str],
    correct_answer: str,
    answer_type: str,
    coefficients: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute reward based on the comprehensive formula:
    R = λ1 * C_logic + λ2 * S_step - λ3 * H_halluc + λ4 * A_correct
    
    Args:
        question: Original question
        generated_output: Dictionary with thinking, verification, conclusion, raw_text
        correct_answer: Reference correct answer
        answer_type: Type of answer ("exactMatch", "multipleChoice", etc.)
        coefficients: Optional custom weight coefficients
        
    Returns:
        Reward value (-1.0 to 1.5)
    """
    # Use default coefficients if none provided
    if coefficients is None:
        coefficients = DEFAULT_REWARD_COEFFICIENTS
        
    # Extract components
    thinking_text = generated_output.get("thinking", "")
    verification_text = generated_output.get("verification", "")
    conclusion_text = generated_output.get("conclusion", "").strip()

    # Get coefficient weights
    lambda_logic = coefficients.get("lambda_logic", 0.3)
    lambda_step = coefficients.get("lambda_step", 0.3)
    lambda_halluc = coefficients.get("lambda_halluc", 0.1)
    lambda_correct = coefficients.get("lambda_wrong", 0.3)

    # 1. Evaluate logical consistency (C_logic)
    thinking_steps = extract_reasoning_steps(thinking_text)
    reasoning_graph = build_reasoning_graph(thinking_steps)
    
    # Create a mapping of node IDs to text content for semantic analysis
    node_contents = {i: step for i, step in enumerate(thinking_steps)}
    
    consistency_score = evaluate_logical_consistency(reasoning_graph, node_contents)

    # 2. Evaluate stepwise correctness (S_step)
    stepwise_score = evaluate_stepwise_correctness(thinking_steps, verification_text)

    # 3. Detect hallucinations (H_halluc)
    hallucination_score = detect_hallucinations(
        thinking_text + " " + verification_text, question
    )

    # 4. Evaluate answer correctness (A_correct)
    answer_score = evaluate_answer_correctness(
        conclusion_text, correct_answer, answer_type, question
    )

    # Calculate final reward using the formula
    reward = (
        lambda_logic * consistency_score
        + lambda_step * stepwise_score
        - lambda_halluc * hallucination_score
        + lambda_correct * answer_score
    )

    # Log component scores
    logger.debug(f"Component Scores:")
    logger.debug(f"  - Logical Consistency: {consistency_score:.4f}")
    logger.debug(f"  - Stepwise Correctness: {stepwise_score:.4f}")
    logger.debug(f"  - Hallucination Penalty: {hallucination_score:.4f}")
    logger.debug(f"  - Answer Correctness: {answer_score:.4f}")
    logger.info(f"Reward: {reward:.4f}")

    # Clamp reward to reasonable range
    return max(min(reward, 1.5), -1.0) 