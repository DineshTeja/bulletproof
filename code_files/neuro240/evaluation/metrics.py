"""Evaluation metrics for assessing model outputs."""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from openai import OpenAI

from neuro240.utils.config import (
    OPENAI_API_KEY,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REWARD_COEFFICIENTS,
    DEVICE,
)

logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def extract_reasoning_steps(thinking_text: str) -> List[str]:
    """Extract individual reasoning steps from thinking text.
    
    Args:
        thinking_text: Text containing reasoning
        
    Returns:
        List of individual reasoning steps
    """
    if not thinking_text:
        return []
        
    # Split text into sentences and filter out short sentences
    sentences = sent_tokenize(thinking_text)
    steps = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Alternatively, look for numbered steps
    numbered_steps = re.findall(r"\d+\.\s+([^\d]+?)(?=\d+\.|$)", thinking_text)
    if numbered_steps and len(numbered_steps) > 1:
        steps = numbered_steps

    return steps


def build_reasoning_graph(thinking_steps: List[str]) -> nx.DiGraph:
    """Build a directed graph representing the reasoning flow.
    
    Args:
        thinking_steps: List of reasoning steps
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()

    # Add nodes for each reasoning step
    for i, step in enumerate(thinking_steps):
        G.add_node(i, text=step)

    # Add edges between consecutive steps to represent the reasoning flow
    for i in range(len(thinking_steps) - 1):
        G.add_edge(i, i + 1)

    return G


def compute_embedding_similarity(answer1: str, answer2: str) -> float:
    """Compute semantic similarity between two text strings.
    
    Args:
        answer1: First answer string
        answer2: Second answer string
        
    Returns:
        Similarity score (0-1)
    """
    emb1 = embedding_model.encode(answer1, convert_to_tensor=True)
    emb2 = embedding_model.encode(answer2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def evaluate_logical_consistency(
    graph: nx.DiGraph, 
    node_contents: Optional[Dict[int, str]] = None
) -> float:
    """Evaluate logical consistency of a reasoning graph.
    
    Args:
        graph: NetworkX directed graph representing reasoning
        node_contents: Optional dict mapping node IDs to text content
        
    Returns:
        Consistency score (-0.5 to 1.0)
    """
    # Base case checks
    if not graph or len(graph.nodes) == 0:
        return 0.0

    # Initialize scores for different dimensions
    scores = {
        "acyclicity": 0.0,
        "connectivity": 0.0,
        "flow_quality": 0.0,
        "depth_breadth": 0.0,
        "semantic_coherence": 0.0
    }

    # 1. Check for cycles - good reasoning shouldn't loop back on itself
    if not nx.is_directed_acyclic_graph(graph):
        # Count cycles and penalize proportionally
        cycles = list(nx.simple_cycles(graph))
        cycle_penalty = min(0.5, 0.1 * len(cycles))
        scores["acyclicity"] = -cycle_penalty
    else:
        scores["acyclicity"] = 0.2

    # 2. Check graph connectivity
    components = list(nx.weakly_connected_components(graph))
    if len(components) > 1:
        # Smaller penalty for minor disconnections
        largest_component_size = max(len(c) for c in components)
        connectivity_ratio = largest_component_size / len(graph.nodes)
        scores["connectivity"] = 0.2 * connectivity_ratio
    else:
        scores["connectivity"] = 0.2

    # 3. Analyze reasoning flow quality
    if len(graph.nodes) > 1:
        # Examine path lengths
        try:
            if nx.is_directed_acyclic_graph(graph):
                longest_path = nx.dag_longest_path(graph)
                longest_path_length = len(longest_path)
                # Reward deeper reasoning chains
                scores["flow_quality"] = min(0.2, 0.05 * longest_path_length)
            else:
                scores["flow_quality"] = 0.05  # Minimal score for non-DAG
        except Exception as e:
            logger.warning(f"Error calculating flow quality: {e}")
            scores["flow_quality"] = 0.05

        # 4. Assess depth vs breadth balance
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        
        avg_in_degree = sum(in_degrees) / len(graph.nodes) if in_degrees else 0
        avg_out_degree = sum(out_degrees) / len(graph.nodes) if out_degrees else 0
        
        if max(avg_in_degree, avg_out_degree) > 0:
            balance_score = min(avg_in_degree, avg_out_degree) / max(avg_in_degree, avg_out_degree)
            scores["depth_breadth"] = 0.2 * balance_score
        else:
            scores["depth_breadth"] = 0.0

    # 5. Semantic coherence if node contents provided
    if node_contents:
        coherence_scores = []
        
        # Check each edge for semantic coherence
        for source, target in graph.edges():
            if source in node_contents and target in node_contents:
                source_content = node_contents[source]
                target_content = node_contents[target]

                # Calculate semantic similarity
                sim_score = compute_embedding_similarity(source_content, target_content)
                coherence_scores.append(min(sim_score, 0.9))  # Cap at 0.9 to avoid perfect scores
                
        # Average coherence score
        if coherence_scores:
            scores["semantic_coherence"] = 0.2 * (sum(coherence_scores) / len(coherence_scores))

    # Calculate final weighted score
    weights = {
        "acyclicity": 0.3,
        "connectivity": 0.2,
        "flow_quality": 0.2,
        "depth_breadth": 0.2,
        "semantic_coherence": 0.1 if node_contents else 0.0
    }

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    final_score = sum(scores[k] * weights[k] for k in scores)

    # Cap final score between -0.5 and 1.0
    return max(-0.5, min(1.0, final_score)) 