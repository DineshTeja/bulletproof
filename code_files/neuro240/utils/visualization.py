"""Visualization utilities for model performance."""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.path import Path as MatplotlibPath
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.patches import RegularPolygon, Circle

from neuro240.utils.config import DEFAULT_PLOTS_DIR
from neuro240.evaluation.evaluator import ModelEvaluationResult

logger = logging.getLogger(__name__)


def set_plotting_style() -> None:
    """Set default plotting style."""
    plt.style.use('ggplot')
    sns.set_palette("Set2")
    mpl.rcParams['figure.figsize'] = (12, 6)
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['font.size'] = 12


def create_reward_comparison_plot(
    before_result: ModelEvaluationResult,
    after_result: ModelEvaluationResult,
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """Create a bar plot comparing rewards before and after fine-tuning.
    
    Args:
        before_result: Evaluation result from before fine-tuning
        after_result: Evaluation result from after fine-tuning
        output_dir: Directory to save the plot (if None, use default)
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    fig, ax = plt.subplots()
    
    # Prepare data
    labels = ['Before', 'After']
    values = [before_result.overall_reward, after_result.overall_reward]
    
    # Create bar plot
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.6)
    
    # Add labels and title
    ax.set_xlabel('Training Stage')
    ax.set_ylabel('Overall Reward')
    ax.set_title(f'Reward Comparison for {before_result.model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Calculate improvement
    improvement = after_result.overall_reward - before_result.overall_reward
    pct_improvement = (improvement / before_result.overall_reward * 100) if before_result.overall_reward != 0 else 0
    
    # Add improvement text
    ax.text(0.5, 0.9, f'Improvement: {improvement:.4f} ({pct_improvement:.2f}%)',
            ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = DEFAULT_PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, f'{before_result.model_name}_reward_comparison.png')
    plt.savefig(plot_path, dpi=300)
    
    if show_plot:
        plt.show()
    plt.close()
    
    return plot_path


def create_metrics_comparison_plot(
    before_result: ModelEvaluationResult,
    after_result: ModelEvaluationResult,
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """Create a grouped bar plot comparing all metrics before and after fine-tuning.
    
    Args:
        before_result: Evaluation result from before fine-tuning
        after_result: Evaluation result from after fine-tuning
        output_dir: Directory to save the plot (if None, use default)
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    metrics = [
        'logical_consistency',
        'stepwise_correctness',
        'hallucination_penalty',
        'answer_correctness',
        'overall_reward',
    ]
    
    pretty_names = {
        'logical_consistency': 'Logical\nConsistency',
        'stepwise_correctness': 'Stepwise\nCorrectness',
        'hallucination_penalty': 'Hallucination\nPenalty',
        'answer_correctness': 'Answer\nCorrectness',
        'overall_reward': 'Overall\nReward',
    }
    
    before_values = [getattr(before_result, metric) for metric in metrics]
    after_values = [getattr(after_result, metric) for metric in metrics]
    
    # Set up the bar chart
    x = np.arange(len(metrics))  
    width = 0.35
    
    # Create bars
    before_bars = ax.bar(x - width/2, before_values, width, label='Before')
    after_bars = ax.bar(x + width/2, after_values, width, label='After')
    
    # Add labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Metrics Comparison for {before_result.model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_names.get(m, m) for m in metrics])
    ax.legend()
    
    # Add value annotations
    def add_annotations(bars, values):
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_annotations(before_bars, before_values)
    add_annotations(after_bars, after_values)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = DEFAULT_PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, f'{before_result.model_name}_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300)
    
    if show_plot:
        plt.show()
    plt.close()
    
    return plot_path


def radar_factory(num_vars: int, frame: str = 'circle') -> Tuple[np.ndarray, type]:
    """Create a radar chart factory.
    
    Args:
        num_vars: Number of variables for radar chart
        frame: Shape of frame ('circle' or 'polygon')
        
    Returns:
        Tuple of (theta, RadarAxes)
    """
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            self.set_theta_direction(-1)
            self.set_varlabels = self._set_varlabels
        
        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        
        def _set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'
                spine = Spine(axes=self, spine_type='circle',
                               path=MatplotlibPath.unit_regular_polygon(num_vars))
                # unit_regular_polygon returns a polygon with center at (0, 0)
                # and radius 1.
                spine.set_transform(plt.Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    
    # Register custom projection
    register_projection(RadarAxes)
    
    return theta, RadarAxes


def create_radar_chart(
    before_result: ModelEvaluationResult,
    after_result: ModelEvaluationResult,
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """Create a radar chart comparing metrics before and after fine-tuning.
    
    Args:
        before_result: Evaluation result from before fine-tuning
        after_result: Evaluation result from after fine-tuning
        output_dir: Directory to save the plot (if None, use default)
        show_plot: Whether to display the plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    
    # Prepare data
    metrics = [
        'logical_consistency', 
        'stepwise_correctness',
        'answer_correctness',
        'overall_reward',
    ]
    
    # Note that for hallucination_penalty, lower is better, so we'll invert it
    # to make the radar chart consistent (higher is better for all metrics)
    before_values = [
        before_result.logical_consistency,
        before_result.stepwise_correctness,
        before_result.answer_correctness,
        before_result.overall_reward,
    ]
    
    after_values = [
        after_result.logical_consistency,
        after_result.stepwise_correctness,
        after_result.answer_correctness,
        after_result.overall_reward,
    ]
    
    # Create hallucination metrics (inverted)
    metrics.insert(2, 'hallucination_penalty')
    before_values.insert(2, 1.0 - before_result.hallucination_penalty)
    after_values.insert(2, 1.0 - after_result.hallucination_penalty)
    
    # Create prettier labels
    labels = [
        'Logical\nConsistency', 
        'Stepwise\nCorrectness',
        'Hallucination\n(inverted)',
        'Answer\nCorrectness',
        'Overall\nReward',
    ]
    
    # Create the radar chart
    theta, radar_axes = radar_factory(len(metrics), frame='polygon')
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    # Plot the data
    ax.plot(theta, before_values, 'o-', label='Before')
    ax.fill(theta, before_values, alpha=0.25)
    ax.plot(theta, after_values, 'o-', label='After')
    ax.fill(theta, after_values, alpha=0.25)
    
    # Add labels
    ax._set_varlabels(labels)
    
    # Add title and legend
    plt.title(f'Performance Metrics for {before_result.model_name}', size=15, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save plot
    if output_dir is None:
        output_dir = DEFAULT_PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, f'{before_result.model_name}_radar_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    plt.close()
    
    return plot_path 