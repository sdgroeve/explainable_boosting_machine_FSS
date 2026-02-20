"""
Configuration dataclasses and constants for EBM Runner.

This module contains all configuration settings, constants, and dataclasses
used throughout the EBM runner package.
"""

from dataclasses import dataclass, field
from typing import Optional


# Constants
DEFAULT_DPI = 200
DEFAULT_MAX_ROWS = 20
DEFAULT_WRAP_LENGTH = 110
DEFAULT_RANDOM_STATE = 42


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    
    dpi: int = DEFAULT_DPI
    feature_importance_figsize: tuple = (8, 6)
    shape_function_figsize: tuple = (7.5, 3.5)
    local_explanation_figsize: tuple = (8, 4)
    classification_curve_figsize: tuple = (5.5, 4)
    regression_diagnostic_figsize: tuple = (5.5, 4)


@dataclass
class ReportConfig:
    """Configuration for PDF report generation."""
    
    max_rows_in_table: int = DEFAULT_MAX_ROWS
    wrap_length: int = DEFAULT_WRAP_LENGTH
    top_n_features: int = 12
    n_local_samples: int = 3
    images_per_row: int = 2
    image_width_cm: float = 8.0


@dataclass
class RunConfig:
    """Main configuration for EBM runs."""
    
    output_dir: str = "./ebm_output"
    random_state: int = DEFAULT_RANDOM_STATE
    force_task: Optional[str] = None
    test_size: float = 0.2
    val_size: float = 0.0
    tuning_n_splits: int = 5            # CV folds for hyperparameter search
    tuning_stratified: bool = True       # stratify tuning folds (clf only)
    eval_n_splits: int = 5              # CV folds for cv_only evaluation
    eval_stratified: bool = True         # stratify eval folds (clf only)
    n_iter: int = 2
    use_grid: bool = False
    n_jobs: int = 1
    ebm_n_jobs: int = -1
    enable_interactions: bool = True
    eval_strategy: str = "train_only"   # "train_test" | "cv_only" | "train_only"
    tune_once: bool = False             # if True, tune only once on full features
    tuning_verbose: int = 1              # verbosity of hyperparameter search
    tuning_show_all: bool = False        # if True, log all candidates in console
    explain_positive_class: bool = False # if True, add local explanations for all positive-class samples
    save_predictions: bool = False       # if True, write predictions.csv in original dataset order
    
    # Sub-configurations
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    report_config: ReportConfig = field(default_factory=ReportConfig)

