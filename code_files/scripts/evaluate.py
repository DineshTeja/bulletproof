#!/usr/bin/env python
"""Evaluation script for Neuro240 models.

This script evaluates a language model's structured reasoning capabilities.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the src directory to the path so we can import the package
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

from neuro240.utils.config import (
    DEFAULT_MODEL,
    DEFAULT_EVALUATION_CONFIG,
    get_model_path,
    get_output_model_path,
    set_seed,
    validate_API_keys,
)
from neuro240.models.model_setup import setup_model, load_fine_tuned_model
from neuro240.utils.data import load_hle_dataset
from neuro240.evaluation.evaluator import evaluate_model
from neuro240.utils.visualization import (
    create_metrics_comparison_plot,
    create_radar_chart,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a language model's reasoning capabilities")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Name of the model to evaluate (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the fine-tuned model (default: auto)",
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the baseline (non-fine-tuned) model",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_EVALUATION_CONFIG["num_samples"],
        help=f"Number of dataset samples to use (default: {DEFAULT_EVALUATION_CONFIG['num_samples']})",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the evaluation results (default: auto)",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fine-tuned model with baseline",
    )
    
    return parser.parse_args()


def main():
    """Run the evaluation script."""
    args = parse_args()
    
    # Validate API keys
    if not validate_API_keys():
        logger.error("Missing API keys. Please set them in your .env file.")
        sys.exit(1)
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(str(get_output_model_path(args.model_name).parent), "results")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load the dataset
    logger.info(f"Loading dataset with {args.num_samples} samples")
    dataset = load_hle_dataset(num_samples=args.num_samples)
    
    # Evaluate baseline model if requested or doing comparison
    if args.baseline or args.compare:
        logger.info(f"Setting up baseline model: {args.model_name}")
        baseline_tokenizer, baseline_model = setup_model(args.model_name)
        
        logger.info("Evaluating baseline model")
        baseline_result = evaluate_model(
            model=baseline_model,
            tokenizer=baseline_tokenizer,
            eval_dataset=dataset,
            model_name=f"{args.model_name}_baseline",
        )
        
        # Save baseline results
        baseline_output_path = os.path.join(output_dir, f"{args.model_name}_baseline_results.json")
        with open(baseline_output_path, "w") as f:
            import json
            json.dump(baseline_result.to_dict(), f, indent=2)
        
        logger.info(f"Saved baseline results to {baseline_output_path}")
    
    # Skip fine-tuned model evaluation if only evaluating baseline
    if args.baseline and not args.compare:
        logger.info("Baseline evaluation completed successfully")
        return 0
    
    # Determine fine-tuned model path
    if args.model_path:
        fine_tuned_path = args.model_path
    else:
        fine_tuned_path = str(get_output_model_path(args.model_name))
    
    # Check if fine-tuned model exists
    if not os.path.exists(fine_tuned_path):
        logger.error(f"Fine-tuned model not found at {fine_tuned_path}")
        sys.exit(1)
    
    # Load and evaluate fine-tuned model
    logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
    fine_tuned_tokenizer, fine_tuned_model = load_fine_tuned_model(fine_tuned_path)
    
    logger.info("Evaluating fine-tuned model")
    fine_tuned_result = evaluate_model(
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        eval_dataset=dataset,
        model_name=f"{args.model_name}_fine_tuned",
    )
    
    # Save fine-tuned results
    fine_tuned_output_path = os.path.join(output_dir, f"{args.model_name}_fine_tuned_results.json")
    with open(fine_tuned_output_path, "w") as f:
        import json
        json.dump(fine_tuned_result.to_dict(), f, indent=2)
    
    logger.info(f"Saved fine-tuned results to {fine_tuned_output_path}")
    
    # Create comparison if requested
    if args.compare:
        logger.info("Creating comparison visualizations")
        
        # Create plots directory
        plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate comparison plots
        comparison_df = create_comparison_dataframe(baseline_result, fine_tuned_result)
        comparison_df_path = os.path.join(output_dir, f"{args.model_name}_comparison.csv")
        comparison_df.to_csv(comparison_df_path, index=False)
        
        # Create visualizations
        metrics_plot_path = create_metrics_comparison_plot(baseline_result, fine_tuned_result, plots_dir)
        radar_plot_path = create_radar_chart(baseline_result, fine_tuned_result, plots_dir)
        
        logger.info(f"Saved comparison results to {comparison_df_path}")
        logger.info(f"Saved comparison plots to {plots_dir}")
    
    logger.info("Evaluation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 