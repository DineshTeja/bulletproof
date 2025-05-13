#!/usr/bin/env python
"""Training script for Neuro240 models.

This script trains a language model using PPO to enhance structured reasoning.
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
    DEFAULT_TRAINING_CONFIG,
    get_model_path,
    get_output_model_path,
    set_seed,
    validate_API_keys,
)
from neuro240.models.model_setup import setup_model, save_model
from neuro240.utils.data import load_hle_dataset, create_splits
from neuro240.training.ppo import train_with_ppo
from neuro240.evaluation.evaluator import evaluate_model, create_comparison_dataframe
from neuro240.utils.visualization import (
    create_reward_comparison_plot,
    create_metrics_comparison_plot,
    create_radar_chart,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a language model with PPO")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Name of the model to train (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["num_epochs"],
        help=f"Number of training epochs (default: {DEFAULT_TRAINING_CONFIG['num_epochs']})",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["batch_size"],
        help=f"Batch size for training (default: {DEFAULT_TRAINING_CONFIG['batch_size']})",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_TRAINING_CONFIG["learning_rate"],
        help=f"Learning rate (default: {DEFAULT_TRAINING_CONFIG['learning_rate']})",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of dataset samples to use (default: 50)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["seed"],
        help=f"Random seed (default: {DEFAULT_TRAINING_CONFIG['seed']})",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the model (default: auto)",
    )
    
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation and only do training",
    )
    
    return parser.parse_args()


def main():
    """Run the training script."""
    args = parse_args()
    
    # Validate API keys
    if not validate_API_keys():
        logger.error("Missing API keys. Please set them in your .env file.")
        sys.exit(1)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create a training configuration from arguments
    training_config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(get_output_model_path(args.model_name))
        
    logger.info(f"Training config: {training_config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load the dataset
    logger.info(f"Loading dataset with {args.num_samples} samples")
    dataset = load_hle_dataset(num_samples=args.num_samples)
    
    # Create train/val/test splits
    train_dataset, val_dataset, test_dataset = create_splits(
        dataset,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
    )
    
    # Set up the model
    logger.info(f"Setting up model: {args.model_name}")
    tokenizer, model = setup_model(args.model_name)
    
    # Evaluate the model before training if not skipping eval
    if not args.skip_eval:
        logger.info("Evaluating model before training")
        before_result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            max_samples=min(10, len(test_dataset)),  # Limit evaluation samples
        )
    
    # Train the model
    logger.info("Starting training")
    trained_model = train_with_ppo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=training_config,
    )
    
    # Save the trained model
    logger.info(f"Saving trained model to {output_dir}")
    save_model(trained_model, tokenizer, output_dir)
    
    # Evaluate the model after training if not skipping eval
    if not args.skip_eval:
        logger.info("Evaluating model after training")
        after_result = evaluate_model(
            model=trained_model,
            tokenizer=tokenizer,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            max_samples=min(10, len(test_dataset)),  # Limit evaluation samples
        )
        
        # Create comparison dataframe
        comparison_df = create_comparison_dataframe(before_result, after_result)
        
        # Save comparison results
        results_dir = os.path.join(os.path.dirname(output_dir), "results")
        os.makedirs(results_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(results_dir, f"{args.model_name}_comparison.csv"), index=False)
        
        # Create visualizations
        plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        reward_plot_path = create_reward_comparison_plot(before_result, after_result, plots_dir)
        metrics_plot_path = create_metrics_comparison_plot(before_result, after_result, plots_dir)
        radar_plot_path = create_radar_chart(before_result, after_result, plots_dir)
        
        logger.info(f"Saved comparison results to {os.path.join(results_dir, f'{args.model_name}_comparison.csv')}")
        logger.info(f"Saved plots to {plots_dir}")
    
    logger.info("Training completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 