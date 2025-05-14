"""Evaluation of answer correctness."""

import logging
import json
from typing import Dict, Any, Optional

from openai import OpenAI

from neuro240.utils.config import OPENAI_API_KEY, DEFAULT_OPENAI_MODEL
from neuro240.evaluation.metrics import compute_embedding_similarity

logger = logging.getLogger(__name__)


def evaluate_answer_correctness(
    extracted_answer: str,
    correct_answer: str,
    answer_type: str,
    question: str
) -> float:
    """Evaluate answer correctness using an LLM-based approach with nuanced scoring.
    
    Args:
        extracted_answer: Model's answer to evaluate
        correct_answer: Reference correct answer
        answer_type: Type of answer ("exactMatch", "multipleChoice", etc.)
        question: Original question
        
    Returns:
        Correctness score (0-1)
    """
    if not extracted_answer:
        return 0.0

    # For multiple choice questions, we can do a simple check
    if answer_type == "multipleChoice" and extracted_answer.upper().strip() == correct_answer.upper().strip():
        return 1.0
        
    # Always use LLM-based evaluation
    try:
        return evaluate_with_llm(extracted_answer, correct_answer, answer_type, question)
    except Exception as e:
        logger.error(f"Error evaluating answer correctness with LLM: {e}")
        # Fall back to embedding similarity
        return compute_embedding_similarity(extracted_answer, correct_answer) * 0.7


def evaluate_with_llm(
    extracted_answer: str,
    correct_answer: str,
    answer_type: str,
    question: str
) -> float:
    """Evaluate answer correctness using an LLM.
    
    Args:
        extracted_answer: Model's answer to evaluate
        correct_answer: Reference correct answer
        answer_type: Type of answer ("exactMatch", "multipleChoice", etc.)
        question: Original question
        
    Returns:
        Correctness score (0-1)
        
    Raises:
        ValueError: If OpenAI API key is not available
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not available")
        
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Define a structured format for evaluation
    system_prompt = """You are an expert answer evaluation assistant specializing in nuanced correctness assessment.

    SCORING GUIDELINES:
    - For COMPLETELY CORRECT answers: is_correct=true, confidence between 0.8-1.0
    - For MOSTLY CORRECT answers with minor issues: is_correct=true, confidence between 0.6-0.8
    - For PARTIALLY CORRECT answers with significant gaps: is_correct=false, confidence between 0.3-0.6
    - For MOSTLY INCORRECT answers with some valid elements: is_correct=false, confidence between 0.1-0.3
    - For COMPLETELY INCORRECT answers: is_correct=false, confidence between 0.0-0.1

    Use the FULL RANGE of confidence scores within each category. A score of 0.0 should be reserved only for answers that are completely unrelated or nonsensical.

    Return a JSON with these fields:
    - "is_correct": A boolean value (true/false) indicating if the answer is correct overall
    - "confidence": A precise float between 0.0 and 1.0 following the guidelines above
    - "correctness_category": One of ["COMPLETELY CORRECT", "MOSTLY CORRECT", "PARTIALLY CORRECT", "MOSTLY INCORRECT", "COMPLETELY INCORRECT"]
    - "explanation": Your detailed reasoning for this assessment, including what was right and wrong
    """

    user_prompt = f"""
    Question: {question}

    Candidate Answer: {extracted_answer}

    Reference Answer: {correct_answer}

    Answer Type: {answer_type}

    EVALUATION TASK:
    1. Analyze both the candidate answer and reference answer in relation to the question
    2. For multiple choice questions, check if the selected option matches, even if explanations differ
    3. For exact match questions, evaluate semantic equivalence and conceptual accuracy
    4. Identify any partial correctness or elements of truth in incorrect answers
    5. Apply the scoring guidelines to determine the appropriate confidence score
    6. Classify the answer into one of the correctness categories
    7. Provide a detailed explanation justifying your assessment

    BE DELIBERATE: Ensure your confidence score precisely reflects the degree of correctness - avoid defaulting to standard values.
    """

    try:
        response = openai_client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,  # Slight temperature to allow variation
        )

        result = json.loads(response.choices[0].message.content)
        is_correct = result.get("is_correct", False)
        confidence = result.get("confidence", 0.0)
        category = result.get("correctness_category", "UNKNOWN")

        # Log the assessment for debugging
        logger.info(f"Answer evaluation: {category}, Confidence: {confidence:.2f}")
        logger.debug(f"Explanation: {result.get('explanation', 'No explanation provided')[:100]}...")

        # Calculate final score based on correctness category and confidence
        if is_correct:
            return confidence
        else:
            # Scale partial credit based on category
            if category == "PARTIALLY CORRECT":
                return confidence * 0.5  # Higher scaling for partially correct
            elif category == "MOSTLY INCORRECT":
                return confidence * 0.3  # Some credit for mostly incorrect
            else:
                return confidence * 0.1  # Minimal credit for completely incorrect

    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        raise


def evaluate_stepwise_correctness(thinking_steps: list, verification_text: str) -> float:
    """Evaluate the correctness of reasoning steps.
    
    Args:
        thinking_steps: List of reasoning steps
        verification_text: Verification text that should reference the steps
        
    Returns:
        Stepwise correctness score (0-1)
    """
    import re
    from nltk.corpus import stopwords
    
    if not thinking_steps:
        return 0.0

    try:
        stop_words = set(stopwords.words("english"))
    except:
        # If NLTK data isn't available, use a simple list
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through"}

    # Basic check: verify that verification references the thinking steps
    step_references = 0
    for step in thinking_steps:
        # Create a simplified version of the step for comparison
        step_keywords = set(
            word.lower()
            for word in re.findall(r"\b\w+\b", step)
            if word.lower() not in stop_words and len(word) > 3
        )

        # Check if verification text contains keywords from this step
        verification_words = set(
            word.lower()
            for word in re.findall(r"\b\w+\b", verification_text)
            if len(word) > 3
        )

        common_keywords = step_keywords.intersection(verification_words)
        if len(common_keywords) >= min(2, len(step_keywords) // 3):
            step_references += 1

    # Calculate score based on the portion of steps that were verified
    if len(thinking_steps) > 0:
        verification_ratio = step_references / len(thinking_steps)
        # Base score of 0.3 even if no steps verified, up to 0.9 if all steps verified
        return min(0.9, 0.3 + 0.6 * verification_ratio)
        
    return 0.3  # Base score if steps exist but none are verified 