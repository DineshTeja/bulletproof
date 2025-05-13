"""Hallucination detection and evaluation."""

import re
import logging
from typing import Set, List, Dict, Any

import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import util

from neuro240.evaluation.metrics import embedding_model

logger = logging.getLogger(__name__)


def detect_hallucinations(reasoning_text: str, question: str, threshold: float = 0.4) -> float:
    """Enhanced heuristic for detecting potential hallucinations.
    
    Args:
        reasoning_text: Text to analyze for hallucinations
        question: Original question for context
        threshold: Similarity threshold below which content is considered irrelevant
        
    Returns:
        Hallucination score (0-0.8), where higher indicates more hallucination
    """
    if not reasoning_text:
        return 0.0

    # 1. Pattern-based detection with expanded patterns
    hallucination_patterns = [
        # Overconfidence patterns
        r"it is (well-known|widely accepted|obviously|clearly|certainly|definitely) that",
        r"(everyone|all experts|all scientists|all mathematicians) (knows|agree|accept)",
        r"(without a doubt|undoubtedly|unquestionably|absolutely|certainly)",
        r"(the only possible|the sole|the exclusive|the definitive) (answer|solution|approach)",

        # Citation hallucinations
        r"(according to|as stated in|as shown in) (research|studies|papers|experiments|literature)",
        r"(studies|research|papers|experts|scientists) (have shown|have proven|have demonstrated|have found)",

        # Specific claim patterns
        r"(exactly|precisely) (the same as|equal to|equivalent to)",
        r"(always|never|in all cases|in every instance|without exception)",

        # Number hallucinations
        r"(\d+) (percent|per cent|%)",
        r"in (\d{4}), (.*) (discovered|invented|found|proven)",
    ]

    # 2. Calculate relevance of reasoning to question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    reasoning_embedding = embedding_model.encode(reasoning_text, convert_to_tensor=True)
    relevance_score = util.pytorch_cos_sim(question_embedding, reasoning_embedding).item()

    # If reasoning seems irrelevant to question, increase hallucination score
    irrelevance_penalty = max(0, threshold - relevance_score) * 2.0

    # 3. Extract statements and check each one
    sentences = sent_tokenize(reasoning_text)
    statements_score = 0.0

    # Set of key terms from question to check against
    question_words = set(re.findall(r'\b[A-Za-z]{4,}\b', question.lower()))

    for sentence in sentences:
        # Skip short sentences
        if len(sentence.split()) < 4:
            continue

        # Check for specific patterns
        pattern_detected = False
        for pattern in hallucination_patterns:
            if re.search(pattern, sentence.lower()):
                statements_score += 0.15
                pattern_detected = True
                break

        # If no pattern detected, check for statement disconnection
        if not pattern_detected:
            # Look for sentences that introduce new concepts not in question
            sentence_words = set(re.findall(r'\b[A-Za-z]{4,}\b', sentence.lower()))
            new_terms = sentence_words - question_words

            # If sentence introduces many new terms AND uses strong language
            if len(new_terms) > 3 and any(
                intensifier in sentence.lower() for intensifier in
                ["very", "extremely", "highly", "completely", "truly"]
            ):
                statements_score += 0.1

    # 4. Check for numerical consistency
    numbers_in_question = set(re.findall(r'\b\d+\b', question))
    numbers_in_reasoning = set(re.findall(r'\b\d+\b', reasoning_text))
    new_numbers = numbers_in_reasoning - numbers_in_question

    # If reasoning introduces many new numbers not in question
    numbers_penalty = min(0.2, len(new_numbers) * 0.05)

    # 5. Final hallucination score calculation
    pattern_score = min(0.5, statements_score)
    hallucination_score = pattern_score + irrelevance_penalty + numbers_penalty

    # Cap and return the hallucination penalty
    return min(0.8, hallucination_score)


def extract_factual_claims(text: str) -> List[str]:
    """Extract factual claims from reasoning text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of extracted factual claims
    """
    if not text:
        return []
        
    # Get sentences first
    sentences = sent_tokenize(text)
    
    # Patterns that often indicate factual claims
    claim_patterns = [
        r"(is|are|was|were) (a|an|the)",
        r"(has|have|had) (a|an|the)",
        r"(equals|equal to|equivalent to|the same as)",
        r"(greater than|less than|more than|fewer than)",
        r"(causes|caused by|results in|leads to)",
        r"(means|implies|indicates|suggests)",
        r"(discovered|invented|found|established|created|developed)",
        r"(known as|referred to as|called)",
    ]
    
    factual_claims = []
    
    for sentence in sentences:
        # Skip questions - they're not claims
        if sentence.strip().endswith("?"):
            continue
            
        # Skip short sentences
        if len(sentence.split()) < 5:
            continue
        
        # Check if sentence contains claim patterns
        for pattern in claim_patterns:
            if re.search(pattern, sentence.lower()):
                factual_claims.append(sentence.strip())
                break
    
    return factual_claims 