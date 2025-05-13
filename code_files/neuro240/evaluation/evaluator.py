"""Model evaluation utilities."""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from neuro240.utils.config import DEVICE, DEFAULT_EVALUATION_CONFIG
from neuro240.utils.types import MetricsResult
from neuro240.models.generation import generate_text
from neuro240.evaluation.reward import compute_reward

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Results from evaluating a model."""
    
    # Model info
    model_name: str
    
    # Performance metrics
    overall_reward: float
    logical_consistency: float
    stepwise_correctness: float
    hallucination_penalty: float
    answer_correctness: float
    
    # Additional stats
    num_processed: int
    num_skipped: int
    total_questions: int
    
    # Correctness
    correct_answers: int  # Number of correct answers (with threshold >= 0.7)
    binary_accuracy: float  # correct_answers / num_processed
    
    # Time
    evaluation_time: float  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "overall_reward": self.overall_reward,
            "logical_consistency": self.logical_consistency,
            "stepwise_correctness": self.stepwise_correctness,
            "hallucination_penalty": self.hallucination_penalty,
            "answer_correctness": self.answer_correctness,
            "num_processed": self.num_processed,
            "num_skipped": self.num_skipped,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "binary_accuracy": self.binary_accuracy,
            "evaluation_time": self.evaluation_time,
        }


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    model_name: str,
    max_samples: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ModelEvaluationResult:
    """Evaluate model performance on a dataset.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset to evaluate on
        model_name: Name of the model being evaluated
        max_samples: Maximum number of samples to evaluate (None for all)
        config: Evaluation configuration

    Returns:
        ModelEvaluationResult with evaluation results
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_EVALUATION_CONFIG

    # Use model's device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Start timing
    start_time = time.time()
    
    # Collect metrics
    rewards = []
    logical_consistency_scores = []
    stepwise_correctness_scores = []
    hallucination_scores = []
    answer_correctness_scores = []
    
    processed_questions = 0
    skipped_questions = 0
    
    # Limit number of samples if specified
    total_questions = len(eval_dataset)
    if max_samples is not None and max_samples < total_questions:
        indices = list(range(total_questions))
        import random
        random.shuffle(indices)
        eval_indices = indices[:max_samples]
        total_questions = max_samples
    else:
        eval_indices = list(range(total_questions))

    logger.info(f"Evaluating {model_name} on {total_questions} questions")

    for i, idx in enumerate(eval_indices):
        try:
            logger.info(f"Processing question {i+1}/{total_questions}")
            item = eval_dataset[idx]
            
            question = item["question"]
            answer = item["answer"]
            answer_type = item["answer_type"]

            # Generate text with error handling
            generated_output = generate_text(
                model, 
                tokenizer, 
                question, 
                answer_type,
                max_length=config.get("max_generation_length", 150),
                temperature=config.get("temperature", 0.7),
                device=device
            )

            # Skip if generation failed completely
            if generated_output is None:
                logger.warning(f"Skipping question {i+1} due to generation failure")
                skipped_questions += 1
                continue

            # Extract components for logging
            thinking = generated_output.get("thinking", "")
            verification = generated_output.get("verification", "")
            conclusion = generated_output.get("conclusion", "")
            
            logger.info(f"Generated answer: '{conclusion}'")
            logger.info(f"Reference answer: '{answer}'")

            # Calculate reward and component scores
            from neuro240.evaluation.metrics import (
                extract_reasoning_steps,
                build_reasoning_graph, 
                evaluate_logical_consistency
            )
            from neuro240.evaluation.correctness import (
                evaluate_answer_correctness,
                evaluate_stepwise_correctness
            )
            from neuro240.evaluation.hallucination import detect_hallucinations
            
            # Get thinking steps and build graph
            thinking_steps = extract_reasoning_steps(thinking)
            reasoning_graph = build_reasoning_graph(thinking_steps)
            node_contents = {i: step for i, step in enumerate(thinking_steps)}
            
            # Calculate component scores
            logical_consistency = evaluate_logical_consistency(reasoning_graph, node_contents)
            stepwise_correctness = evaluate_stepwise_correctness(thinking_steps, verification)
            hallucination_penalty = detect_hallucinations(thinking + " " + verification, question)
            answer_correctness = evaluate_answer_correctness(conclusion, answer, answer_type, question)
            
            # Calculate reward
            reward = compute_reward(question, generated_output, answer, answer_type)

            # Store metrics
            rewards.append(reward)
            logical_consistency_scores.append(logical_consistency)
            stepwise_correctness_scores.append(stepwise_correctness)
            hallucination_scores.append(hallucination_penalty)
            answer_correctness_scores.append(answer_correctness)
            
            processed_questions += 1

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            skipped_questions += 1
            continue

    # Calculate metrics
    if processed_questions == 0:
        logger.error("No questions were successfully processed")
        # Return default metrics
        return ModelEvaluationResult(
            model_name=model_name,
            overall_reward=0.0,
            logical_consistency=0.0,
            stepwise_correctness=0.0,
            hallucination_penalty=0.0,
            answer_correctness=0.0,
            num_processed=0,
            num_skipped=skipped_questions,
            total_questions=total_questions,
            correct_answers=0,
            binary_accuracy=0.0,
            evaluation_time=time.time() - start_time,
        )
    
    # Calculate average scores
    avg_reward = sum(rewards) / processed_questions
    avg_logical_consistency = sum(logical_consistency_scores) / processed_questions
    avg_stepwise_correctness = sum(stepwise_correctness_scores) / processed_questions
    avg_hallucination = sum(hallucination_scores) / processed_questions
    avg_answer_correctness = sum(answer_correctness_scores) / processed_questions
    
    # Calculate binary accuracy (answers above threshold)
    threshold = 0.7
    correct_answers = sum(1 for score in answer_correctness_scores if score >= threshold)
    binary_accuracy = correct_answers / processed_questions
    
    # End timing
    evaluation_time = time.time() - start_time
    
    # Create and return result
    result = ModelEvaluationResult(
        model_name=model_name,
        overall_reward=avg_reward,
        logical_consistency=avg_logical_consistency,
        stepwise_correctness=avg_stepwise_correctness,
        hallucination_penalty=avg_hallucination,
        answer_correctness=avg_answer_correctness,
        num_processed=processed_questions,
        num_skipped=skipped_questions,
        total_questions=total_questions,
        correct_answers=correct_answers,
        binary_accuracy=binary_accuracy,
        evaluation_time=evaluation_time,
    )
    
    # Log summary
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Processed {processed_questions}/{total_questions} questions (skipped {skipped_questions})")
    logger.info(f"Average reward: {avg_reward:.4f}")
    logger.info(f"Average logical consistency: {avg_logical_consistency:.4f}")
    logger.info(f"Average stepwise correctness: {avg_stepwise_correctness:.4f}")
    logger.info(f"Average hallucination penalty: {avg_hallucination:.4f}")
    logger.info(f"Average answer correctness: {avg_answer_correctness:.4f}")
    logger.info(f"Binary accuracy @ {threshold}: {correct_answers}/{processed_questions} ({binary_accuracy:.4f})")
    
    return result


def create_comparison_dataframe(
    before_result: ModelEvaluationResult,
    after_result: ModelEvaluationResult,
) -> pd.DataFrame:
    """Create a DataFrame comparing before and after evaluation results.
    
    Args:
        before_result: Evaluation result from before fine-tuning
        after_result: Evaluation result from after fine-tuning
        
    Returns:
        DataFrame with comparison metrics
    """
    # Get metrics
    metrics = [
        "logical_consistency",
        "stepwise_correctness",
        "hallucination_penalty",
        "answer_correctness",
        "overall_reward",
        "binary_accuracy",
    ]
    
    rows = []
    
    for metric in metrics:
        before_value = getattr(before_result, metric)
        after_value = getattr(after_result, metric)
        abs_change = after_value - before_value
        
        # Calculate percent change (handle division by zero)
        if before_value != 0:
            pct_change = (abs_change / before_value) * 100
        else:
            pct_change = float('inf') if abs_change > 0 else 0
            
        rows.append({
            "Metric": metric,
            "Before": before_value,
            "After": after_value,
            "Absolute Change": abs_change,
            "Percent Change": pct_change,
        })
    
    return pd.DataFrame(rows) 