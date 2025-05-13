"""Type definitions for Neuro240."""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pydantic import BaseModel


class ExtractedResponse(BaseModel):
    """Structured response extracted from model outputs."""
    thinking: str
    verification: str
    conclusion: str
    raw_text: str


class ModelOutput(BaseModel):
    """Structured output from a model generation."""
    thinking: Optional[str] = None
    verification: Optional[str] = None
    conclusion: Optional[str] = None
    raw_text: str


class MetricsResult(BaseModel):
    """Results of evaluation metrics calculation."""
    logical_consistency: float
    stepwise_correctness: float
    hallucination_penalty: float
    answer_correctness: float
    overall_reward: float


class EvaluationResult(BaseModel):
    """Results of model evaluation."""
    model_name: str
    metrics_before: MetricsResult
    metrics_after: MetricsResult
    
    def get_improvement(self) -> Dict[str, float]:
        """Calculate improvement in metrics."""
        return {
            "logical_consistency": self.metrics_after.logical_consistency - self.metrics_before.logical_consistency,
            "stepwise_correctness": self.metrics_after.stepwise_correctness - self.metrics_before.stepwise_correctness,
            "hallucination_penalty": self.metrics_after.hallucination_penalty - self.metrics_before.hallucination_penalty,
            "answer_correctness": self.metrics_after.answer_correctness - self.metrics_before.answer_correctness,
            "overall_reward": self.metrics_after.overall_reward - self.metrics_before.overall_reward,
        }


class DatasetItem(BaseModel):
    """Item from the HLE dataset."""
    question: str
    answer: str
    answer_type: str


TrainingConfig = Dict[str, Any]
EvaluationConfig = Dict[str, Any]
ModelConfig = Dict[str, Any] 