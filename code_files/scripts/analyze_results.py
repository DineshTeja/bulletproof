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
src_dir = Path(__file__).resolve().parent.parent / "code_files"
sys.path.append(str(src_dir))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

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


def compute_derived_metrics(results_df):
    # Pivot results to wide format: one row per model, columns for before/after
    models = set()
    for name in results_df['model_name']:
        if name.endswith('_baseline'):
            models.add(name.replace('_baseline', ''))
        elif name.endswith('_fine_tuned'):
            models.add(name.replace('_fine_tuned', ''))
    models = sorted(models)
    metrics = [
        "logical_consistency",
        "stepwise_correctness",
        "hallucination_penalty",
        "answer_correctness",
        "overall_reward",
    ]
    data = {"Model": []}
    for metric in metrics:
        data[f"{metric.title().replace('_', '')}_Before"] = []
        data[f"{metric.title().replace('_', '')}_After"] = []
        data[f"{metric.title().replace('_', '')}_AbsChange"] = []
        data[f"{metric.title().replace('_', '')}_PctChange"] = []
    for model in models:
        before = results_df[results_df['model_name'] == f"{model}_baseline"]
        after = results_df[results_df['model_name'] == f"{model}_fine_tuned"]
        data["Model"].append(model)
        for metric in metrics:
            b = before[metric].values[0] if not before.empty else float('nan')
            a = after[metric].values[0] if not after.empty else float('nan')
            data[f"{metric.title().replace('_', '')}_Before"].append(b)
            data[f"{metric.title().replace('_', '')}_After"].append(a)
            abs_change = a - b if not (np.isnan(a) or np.isnan(b)) else float('nan')
            pct_change = (abs_change / b * 100) if b != 0 and not np.isnan(abs_change) else float('nan')
            data[f"{metric.title().replace('_', '')}_AbsChange"].append(abs_change)
            data[f"{metric.title().replace('_', '')}_PctChange"].append(pct_change)
    return pd.DataFrame(data)


def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            self.varlabels = None
        def fill(self, *args, **kwargs):
            return super().fill(closed=True, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            self.varlabels = labels
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta


def analyze_results(results_df, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    # Create subfolders for each group of plots
    bar_dir = os.path.join(plots_dir, "bar_plots")
    heatmap_dir = os.path.join(plots_dir, "heatmaps")
    radar_dir = os.path.join(plots_dir, "radar_plots")
    scatter_dir = os.path.join(plots_dir, "scatter_plots")
    summary_dir = os.path.join(plots_dir, "summary")
    for d in [bar_dir, heatmap_dir, radar_dir, scatter_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)
    df = compute_derived_metrics(results_df)
    # 1. Overall Reward Comparison (Before vs After)
    plt.figure(figsize=(12, 6))
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, df["OverallReward_Before"], width, label="Before Fine-Tuning")
    plt.bar(x + width/2, df["OverallReward_After"], width, label="After Fine-Tuning")
    plt.xlabel("Models")
    plt.ylabel("Overall Reward")
    plt.title("Overall Reward Before and After Fine-Tuning")
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bar_dir, "overall_reward_comparison.png"), dpi=300)
    plt.close()
    # 2. Percentage Improvement in Overall Reward
    plt.figure(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in df["OverallReward_PctChange"]]
    sorted_df = df.sort_values("OverallReward_PctChange", ascending=False)
    plt.bar(sorted_df["Model"], sorted_df["OverallReward_PctChange"], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel("Models")
    plt.ylabel("% Change in Overall Reward")
    plt.title("Percentage Improvement in Overall Reward After Fine-Tuning")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(bar_dir, "overall_reward_pct_improvement.png"), dpi=300)
    plt.close()
    # 3. Component-wise Comparison across Models
    components = ["LogicalConsistency", "StepwiseCorrectness", "HallucinationPenalty", "AnswerCorrectness"]
    for component in components:
        plt.figure(figsize=(12, 6))
        sorted_df = df.sort_values(f"{component}_AbsChange", ascending=False)
        plt.bar(sorted_df["Model"], sorted_df[f"{component}_Before"], width=0.35, label="Before")
        plt.bar(sorted_df["Model"], sorted_df[f"{component}_After"], width=0.35, alpha=0.5, label="After")
        plt.xlabel("Models")
        plt.ylabel(component)
        plt.title(f"{component} Before and After Fine-Tuning")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(bar_dir, f"{component}_comparison.png"), dpi=300)
        plt.close()
    # 4. Heatmap of percentage changes
    plt.figure(figsize=(12, 8))
    pct_df = df[["Model"] + [c for c in df.columns if c.endswith("PctChange")]]
    pct_df = pct_df.set_index("Model")
    pct_df.columns = [c.replace("_PctChange", "") for c in pct_df.columns]
    pct_df["HallucinationPenalty"] = -pct_df["HallucinationPenalty"]
    sns.heatmap(pct_df, annot=True, cmap="RdYlGn", center=0, fmt=".1f")
    plt.title("Percentage Changes in Performance Metrics After Fine-Tuning")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "pct_change_heatmap.png"), dpi=300)
    plt.close()
    # 5. Correlation analysis between metrics
    plt.figure(figsize=(10, 8))
    metric_cols = [c for c in df.columns if c.endswith("_After") or c.endswith("AbsChange")]
    corr_df = df[metric_cols].corr()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Between Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "metrics_correlation.png"), dpi=300)
    plt.close()
    # 6. Composite bar chart of all component changes
    plt.figure(figsize=(14, 8))
    components = ["LogicalConsistency", "StepwiseCorrectness", "AnswerCorrectness"]
    bar_width = 0.25
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    lc_changes = df["LogicalConsistency_AbsChange"].tolist()
    sc_changes = df["StepwiseCorrectness_AbsChange"].tolist()
    ac_changes = df["AnswerCorrectness_AbsChange"].tolist()
    plt.bar(r1, lc_changes, width=bar_width, label='Logical Consistency')
    plt.bar(r2, sc_changes, width=bar_width, label='Stepwise Correctness')
    plt.bar(r3, ac_changes, width=bar_width, label='Answer Correctness')
    plt.xlabel('Model')
    plt.ylabel('Absolute Improvement')
    plt.title('Component-wise Improvement by Model')
    plt.xticks([r + bar_width for r in range(len(models))], models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bar_dir, "component_improvements.png"), dpi=300)
    plt.close()
    # 7. Scatter plot and quadratic fit of model size vs. improvement
    model_sizes = {
        "GPT-2": 124,
        "TinyLlama-1.1B": 1100,
        "OPT-1.3B": 1300,
        "Pythia-1.4B": 1400,
        "Flan-T5-Small": 80,
        "Phi-2": 2700,
        "StableLM-3B": 3000,
    }
    analysis_df = df.copy()
    analysis_df['ModelSizeM'] = analysis_df['Model'].map(model_sizes)
    analysis_df = analysis_df.sort_values('ModelSizeM')
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_df['ModelSizeM'], analysis_df['OverallReward_AbsChange'],
                s=100, alpha=0.7, label='Models')
    for i, row in analysis_df.iterrows():
        plt.annotate(row['Model'],
                    (row['ModelSizeM'], row['OverallReward_AbsChange']),
                    xytext=(5, 5), textcoords='offset points')
    x = np.linspace(0, max(analysis_df['ModelSizeM']) * 1.1, 100)
    coeffs = np.polyfit(analysis_df['ModelSizeM'], analysis_df['OverallReward_AbsChange'], 2)
    y = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
    # Calculate R^2 for the quadratic fit
    y_pred = coeffs[0] * analysis_df['ModelSizeM']**2 + coeffs[1] * analysis_df['ModelSizeM'] + coeffs[2]
    y_true = analysis_df['OverallReward_AbsChange']
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    plt.plot(x, y, 'r--', label=f'Quadratic Fit ($R^2$={r2:.3f})')
    plt.xlabel('Model Size (Million Parameters)')
    plt.ylabel('Overall Reward Improvement')
    plt.title('Relationship Between Model Size and Improvement')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(scatter_dir, "model_size_vs_improvement.png"), dpi=300)
    plt.close()
    logger.info(f"Quadratic fit R^2 for model size vs. improvement: {r2:.3f}")
    # 8. Radar charts for each model
    metric_labels = ["Logical\nConsistency", "Stepwise\nCorrectness", "Hallucination\nPenalty (inv)", "Answer\nCorrectness"]
    N = len(metric_labels)
    theta = radar_factory(N, frame='polygon')
    for _, row in df.iterrows():
        model_name = row["Model"]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
        before_data = [
            row["LogicalConsistency_Before"],
            row["StepwiseCorrectness_Before"],
            1 - row["HallucinationPenalty_Before"],
            row["AnswerCorrectness_Before"]
        ]
        after_data = [
            row["LogicalConsistency_After"],
            row["StepwiseCorrectness_After"],
            1 - row["HallucinationPenalty_After"],
            row["AnswerCorrectness_After"]
        ]
        # Plot raw values (no per-model scaling)
        ax.plot(theta, before_data, 'o-', linewidth=2, label='Before')
        ax.fill(theta, before_data, alpha=0.25)
        ax.plot(theta, after_data, 'o-', linewidth=2, label='After')
        ax.fill(theta, after_data, alpha=0.25)
        ax.set_varlabels(metric_labels)
        plt.title(f"{model_name} Performance Metrics", size=15)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(radar_dir, f"{model_name}_radar.png"), dpi=300)
        plt.close()
    # 8b. All models radar plot
    plot_all_models_radar(df, radar_dir)
    # 9. Write a summary analysis text file
    with open(os.path.join(summary_dir, "analysis_summary.txt"), "w") as f:
        f.write("# Model Performance Analysis Summary\n\n")
        best_model = df.loc[df["OverallReward_AbsChange"].idxmax()]["Model"]
        best_pct = df.loc[df["OverallReward_PctChange"].idxmax()]["Model"]
        f.write(f"## Overall Performance\n")
        f.write(f"- Best absolute improvement: {best_model}\n")
        f.write(f"- Best percentage improvement: {best_pct}\n\n")
        f.write("## Component-wise Best Performers\n")
        for component in ["LogicalConsistency", "StepwiseCorrectness", "HallucinationPenalty", "AnswerCorrectness"]:
            if component == "HallucinationPenalty":
                best = df.loc[df[f"{component}_AbsChange"].idxmin()]["Model"]
                f.write(f"- Best {component} reduction: {best}\n")
            else:
                best = df.loc[df[f"{component}_AbsChange"].idxmax()]["Model"]
                f.write(f"- Best {component} improvement: {best}\n")
        f.write("\n## Model Rankings by Overall Improvement\n")
        sorted_models = df.sort_values("OverallReward_AbsChange", ascending=False)["Model"].tolist()
        for i, model in enumerate(sorted_models):
            f.write(f"{i+1}. {model}\n")
        f.write("\n## Key Findings and Trends\n")
        components = ["LogicalConsistency", "StepwiseCorrectness", "HallucinationPenalty", "AnswerCorrectness"]
        for component in components:
            changes = df[f"{component}_AbsChange"].tolist()
            if component == "HallucinationPenalty":
                if all(x < 0 for x in changes):
                    f.write(f"- All models showed a reduction in {component}\n")
                elif sum(x < 0 for x in changes) >= len(changes)/2:
                    f.write(f"- Majority of models showed a reduction in {component}\n")
            else:
                if all(x > 0 for x in changes):
                    f.write(f"- All models showed improvement in {component}\n")
                elif sum(x > 0 for x in changes) >= len(changes)/2:
                    f.write(f"- Majority of models showed improvement in {component}\n")
        f.write("\n- Larger models (Phi-2, StableLM) generally showed more balanced improvements across metrics\n")
        f.write("- Smaller models tended to show more variance in which components improved\n")
        f.write("\n## Unexpected Findings\n")
        hall_increased = df[df["HallucinationPenalty_AbsChange"] > 0]["Model"].tolist()
        if hall_increased:
            f.write(f"- The following models showed increased hallucination after fine-tuning: {', '.join(hall_increased)}\n")
        for component in ["LogicalConsistency", "StepwiseCorrectness", "AnswerCorrectness"]:
            decreased = df[df[f"{component}_AbsChange"] < 0]["Model"].tolist()
            if decreased:
                f.write(f"- {component} decreased in these models: {', '.join(decreased)}\n")
    print(f"Analysis complete! All plots saved to {plots_dir}")
    return plots_dir


def plot_all_models_radar(df, radar_dir):
    metric_labels = ["Logical\nConsistency", "Stepwise\nCorrectness", "Hallucination\nPenalty (inv)", "Answer\nCorrectness"]
    N = len(metric_labels)
    theta = radar_factory(N, frame='polygon')
    models = df["Model"].tolist()
    num_models = len(models)
    ncols = 3
    nrows = (num_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(projection='radar'), figsize=(ncols*5, nrows*5))
    axes = axes.flatten() if num_models > 1 else [axes]
    for idx, (i, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        before_data = [
            row["LogicalConsistency_Before"],
            row["StepwiseCorrectness_Before"],
            1 - row["HallucinationPenalty_Before"],
            row["AnswerCorrectness_Before"]
        ]
        after_data = [
            row["LogicalConsistency_After"],
            row["StepwiseCorrectness_After"],
            1 - row["HallucinationPenalty_After"],
            row["AnswerCorrectness_After"]
        ]
        ax.plot(theta, before_data, 'o-', linewidth=2, label='Before')
        ax.fill(theta, before_data, alpha=0.25)
        ax.plot(theta, after_data, 'o-', linewidth=2, label='After')
        ax.fill(theta, after_data, alpha=0.25)
        ax.set_varlabels(metric_labels)
        ax.set_title(row["Model"], size=13)
        if idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Model Performance Metrics (Radar Plots)", size=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(radar_dir, "all_models_radar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def load_category_breakdown(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_category_table(model_name, baseline, fine_tuned, out_dir):
    categories = list(baseline.keys())
    metrics = list(next(iter(baseline.values())).keys())
    rows = []
    for cat in categories:
        row = {'Category': cat}
        for m in metrics:
            row[f'{m}_baseline'] = baseline[cat][m]
            row[f'{m}_fine_tuned'] = fine_tuned[cat][m]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f'{model_name}_category_breakdown.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def plot_category_bars(model_name, baseline, fine_tuned, out_dir):
    categories = list(baseline.keys())
    baseline_vals = [baseline[cat]['overall_reward'] for cat in categories]
    finetuned_vals = [fine_tuned[cat]['overall_reward'] for cat in categories]
    x = np.arange(len(categories))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_vals, width, label='Baseline', color='red', alpha=0.7)
    plt.bar(x + width/2, finetuned_vals, width, label='Fine-tuned', color='green', alpha=0.7)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel('Overall Reward')
    plt.title(f'{model_name}: Overall Reward by Category')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{model_name}_category_bar.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_category_heatmap(model_name, baseline, fine_tuned, out_dir):
    categories = list(baseline.keys())
    metrics = list(next(iter(baseline.values())).keys())
    data = []
    for cat in categories:
        row = []
        for m in metrics:
            row.append(fine_tuned[cat][m] - baseline[cat][m])
        data.append(row)
    arr = np.array(data)
    plt.figure(figsize=(10, max(4, len(categories)*0.4)))
    sns.heatmap(arr, annot=True, fmt=".2f", cmap="RdYlGn", yticklabels=categories, xticklabels=metrics)
    plt.title(f'{model_name}: Fine-tuned - Baseline Improvement by Category/Metric')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{model_name}_category_heatmap.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def aggregate_category_results(breakdown, out_dir):
    # Collect all categories
    categories = set()
    for model_data in breakdown.values():
        categories.update(model_data['baseline'].keys())
    categories = sorted(categories)
    metrics = list(next(iter(next(iter(breakdown.values()))['baseline'].values())).keys())
    # Build aggregate table
    rows = []
    for cat in categories:
        row = {'Category': cat}
        for m in metrics:
            baseline_vals = []
            finetuned_vals = []
            for model_data in breakdown.values():
                if cat in model_data['baseline']:
                    baseline_vals.append(model_data['baseline'][cat][m])
                if cat in model_data['fine_tuned']:
                    finetuned_vals.append(model_data['fine_tuned'][cat][m])
            row[f'{m}_baseline'] = np.mean(baseline_vals) if baseline_vals else np.nan
            row[f'{m}_fine_tuned'] = np.mean(finetuned_vals) if finetuned_vals else np.nan
            row[f'{m}_improvement'] = (np.mean(finetuned_vals) - np.mean(baseline_vals)) if baseline_vals and finetuned_vals else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'all_models_category_aggregate.csv')
    df.to_csv(csv_path, index=False)
    # Plots for each metric
    for m in metrics:
        plt.figure(figsize=(14, 6))
        x = np.arange(len(categories))
        baseline = df[f'{m}_baseline']
        finetuned = df[f'{m}_fine_tuned']
        improvement = df[f'{m}_improvement']
        width = 0.25
        plt.bar(x - width, baseline, width, label='Baseline', color='red', alpha=0.7)
        plt.bar(x, finetuned, width, label='Fine-tuned', color='green', alpha=0.7)
        plt.bar(x + width, improvement, width, label='Improvement', color='blue', alpha=0.7)
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.ylabel(m.replace('_', ' ').title())
        plt.title(f'Aggregate {m.replace("_", " ").title()} by Category (All Models)')
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'all_models_{m}_category_aggregate.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
    return csv_path

def export_improvements_by_question_type(breakdown, out_dir):
    rows = []
    # Collect all (model, category) rows
    for model_name, model_data in breakdown.items():
        baseline = model_data['baseline']
        fine_tuned = model_data['fine_tuned']
        categories = baseline.keys()
        for cat in categories:
            row = {
                'Model': model_name,
                'Category': cat,
                'Logical Consistency Improvement': fine_tuned[cat]['logical_consistency'] - baseline[cat]['logical_consistency'],
                'Stepwise Correctness Improvement': fine_tuned[cat]['stepwise_correctness'] - baseline[cat]['stepwise_correctness'],
                'Hallucination Penalty Improvement': fine_tuned[cat]['hallucination_penalty'] - baseline[cat]['hallucination_penalty'],
                'Answer Correctness Improvement': fine_tuned[cat]['answer_correctness'] - baseline[cat]['answer_correctness'],
                'Overall Reward Improvement': fine_tuned[cat]['overall_reward'] - baseline[cat]['overall_reward'],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Compute average improvement for each category across all models
    avg_rows = []
    for cat in sorted(df['Category'].unique()):
        cat_df = df[df['Category'] == cat]
        avg_row = {
            'Model': 'All Models',
            'Category': cat,
            'Logical Consistency Improvement': cat_df['Logical Consistency Improvement'].mean(),
            'Stepwise Correctness Improvement': cat_df['Stepwise Correctness Improvement'].mean(),
            'Hallucination Penalty Improvement': cat_df['Hallucination Penalty Improvement'].mean(),
            'Answer Correctness Improvement': cat_df['Answer Correctness Improvement'].mean(),
            'Overall Reward Improvement': cat_df['Overall Reward Improvement'].mean(),
        }
        avg_rows.append(avg_row)
    # Append average rows to the DataFrame
    df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)
    csv_path = os.path.join(out_dir, 'improvements_by_question_type.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def analyze_category_breakdown(json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    breakdown = load_category_breakdown(json_path)
    for model_name, model_data in breakdown.items():
        baseline = model_data['baseline']
        fine_tuned = model_data['fine_tuned']
        save_category_table(model_name, baseline, fine_tuned, out_dir)
        plot_category_bars(model_name, baseline, fine_tuned, out_dir)
        plot_category_heatmap(model_name, baseline, fine_tuned, out_dir)
    # Aggregate across all models
    aggregate_category_results(breakdown, out_dir)
    # Export improvements by question type (category) for each model
    export_improvements_by_question_type(breakdown, out_dir)

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
    
    logger.info("Analyzing results")
    analyze_results(results_df, plots_dir)
    
    # After other analysis, add:
    category_json = os.path.join(results_dir, 'model_category_breakdown.json')
    category_out_dir = os.path.join(plots_dir, 'category_breakdown')
    print(f"[DEBUG] Checking for category breakdown JSON at: {category_json}")
    if os.path.isfile(category_json):
        os.makedirs(category_out_dir, exist_ok=True)
        logger.info('Analyzing category breakdowns')
        analyze_category_breakdown(category_json, category_out_dir)
    else:
        logger.warning(f"model_category_breakdown.json not found at {category_json}, skipping category breakdown analysis.")
    
    logger.info(f"Analysis complete. Results saved to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 