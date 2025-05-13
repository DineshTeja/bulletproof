#!/usr/bin/env python
"""Analysis script for Neuro240 model results.

This script analyzes and visualizes the results of model evaluations.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the path so we can import the package
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from neuro240.utils.config import DEFAULT_PLOTS_DIR, DEFAULT_RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze model evaluation results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing evaluation results (default: auto)",
    )
    
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory to save output plots (default: auto)",
    )
    
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific model names to analyze (default: all available)",
    )
    
    return parser.parse_args()


def load_results(results_dir: str, model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Load evaluation results from JSON files.
    
    Args:
        results_dir: Directory containing results files
        model_names: Optional list of specific model names to load
        
    Returns:
        DataFrame containing evaluation results
    """
    # Find all result files
    result_files = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            model_name = filename.split("_results.json")[0]
            
            # Skip if not in requested model names
            if model_names and not any(model_name.startswith(name) for name in model_names):
                continue
                
            result_files.append(os.path.join(results_dir, filename))
    
    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return pd.DataFrame()
    
    # Load all results
    results = []
    
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def create_model_comparison_plot(results_df: pd.DataFrame, plots_dir: str) -> str:
    """Create a bar plot comparing model performance.
    
    Args:
        results_df: DataFrame with evaluation results
        plots_dir: Directory to save plots
        
    Returns:
        Path to the saved plot
    """
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Extract model names and rewards
    model_names = results_df["model_name"].tolist()
    rewards = results_df["overall_reward"].tolist()
    
    # Sort by reward
    sorted_indices = np.argsort(rewards)[::-1]  # Descending
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_rewards = [rewards[i] for i in sorted_indices]
    
    # Create the bar chart
    colors = ["#2ca02c" if "_fine_tuned" in name else "#d62728" for name in sorted_names]
    
    bars = plt.bar(range(len(sorted_names)), sorted_rewards, color=colors)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha="center", va="bottom", fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Fine-tuned"),
        Patch(facecolor="#d62728", label="Baseline")
    ]
    plt.legend(handles=legend_elements)
    
    # Format the plot
    plt.xlabel("Model")
    plt.ylabel("Overall Reward")
    plt.title("Model Performance Comparison")
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return plot_path


def create_metrics_comparison_table(results_df: pd.DataFrame, plots_dir: str) -> str:
    """Create a CSV table comparing all metrics across models.
    
    Args:
        results_df: DataFrame with evaluation results
        plots_dir: Directory to save output
        
    Returns:
        Path to the saved table
    """
    # Create a more readable version of the dataframe
    metrics = [
        "logical_consistency",
        "stepwise_correctness",
        "hallucination_penalty",
        "answer_correctness",
        "overall_reward",
        "binary_accuracy",
    ]
    
    comparison_data = []
    
    for metric in metrics:
        row_data = {"Metric": metric}
        
        for _, row in results_df.iterrows():
            model_name = row["model_name"]
            row_data[model_name] = row[metric]
        
        comparison_data.append(row_data)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(plots_dir, "metrics_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    return csv_path


def create_improvement_plot(results_df: pd.DataFrame, plots_dir: str) -> str:
    """Create a plot showing improvement from baseline to fine-tuned.
    
    Args:
        results_df: DataFrame with evaluation results
        plots_dir: Directory to save plots
        
    Returns:
        Path to the saved plot
    """
    # Split into baseline and fine-tuned
    baseline_df = results_df[results_df["model_name"].str.contains("_baseline")]
    finetuned_df = results_df[results_df["model_name"].str.contains("_fine_tuned")]
    
    # Only include models that have both baseline and fine-tuned
    baseline_models = set(baseline_df["model_name"].str.split("_baseline").str[0])
    finetuned_models = set(finetuned_df["model_name"].str.split("_fine_tuned").str[0])
    common_models = baseline_models.intersection(finetuned_models)
    
    if not common_models:
        logger.warning("No models have both baseline and fine-tuned results")
        return ""
    
    # Calculate improvements
    improvements = []
    
    for model in common_models:
        baseline_row = baseline_df[baseline_df["model_name"] == f"{model}_baseline"].iloc[0]
        finetuned_row = finetuned_df[finetuned_df["model_name"] == f"{model}_fine_tuned"].iloc[0]
        
        # Calculate improvement for each metric
        improvement = {
            "model": model,
            "logical_consistency": finetuned_row["logical_consistency"] - baseline_row["logical_consistency"],
            "stepwise_correctness": finetuned_row["stepwise_correctness"] - baseline_row["stepwise_correctness"],
            "hallucination_penalty": finetuned_row["hallucination_penalty"] - baseline_row["hallucination_penalty"],
            "answer_correctness": finetuned_row["answer_correctness"] - baseline_row["answer_correctness"],
            "overall_reward": finetuned_row["overall_reward"] - baseline_row["overall_reward"],
            "binary_accuracy": finetuned_row["binary_accuracy"] - baseline_row["binary_accuracy"],
        }
        
        improvements.append(improvement)
    
    # Convert to DataFrame
    improvements_df = pd.DataFrame(improvements)
    
    # Create a bar chart of improvement in overall reward
    plt.figure(figsize=(10, 6))
    
    models = improvements_df["model"].tolist()
    reward_improvements = improvements_df["overall_reward"].tolist()
    
    # Sort by improvement
    sorted_indices = np.argsort(reward_improvements)[::-1]  # Descending
    sorted_models = [models[i] for i in sorted_indices]
    sorted_improvements = [reward_improvements[i] for i in sorted_indices]
    
    # Create color gradient based on improvement value
    colors = ["#2ca02c" if imp > 0 else "#d62728" for imp in sorted_improvements]
    
    # Create the bar chart
    bars = plt.bar(range(len(sorted_models)), sorted_improvements, color=colors)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha="center", va="bottom", fontsize=9)
    
    # Format the plot
    plt.xlabel("Model")
    plt.ylabel("Improvement in Overall Reward")
    plt.title("Fine-tuning Improvement by Model")
    plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha="right")
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "improvement_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Also save the improvement data
    csv_path = os.path.join(plots_dir, "improvements.csv")
    improvements_df.to_csv(csv_path, index=False)
    
    return plot_path


def main():
    """Run the analysis script."""
    args = parse_args()
    
    # Determine directories
    results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    plots_dir = args.plots_dir or DEFAULT_PLOTS_DIR
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load results
    logger.info(f"Loading results from {results_dir}")
    results_df = load_results(results_dir, args.model_names)
    
    if results_df.empty:
        logger.error("No valid results found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(results_df)} result files")
    
    # Create plots
    logger.info("Creating model comparison plot")
    comparison_plot = create_model_comparison_plot(results_df, plots_dir)
    
    logger.info("Creating metrics comparison table")
    metrics_table = create_metrics_comparison_table(results_df, plots_dir)
    
    logger.info("Creating improvement plot")
    improvement_plot = create_improvement_plot(results_df, plots_dir)
    
    logger.info(f"Analysis complete. Results saved to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 