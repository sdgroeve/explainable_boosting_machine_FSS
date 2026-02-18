"""
EBM Runner - Explainable Boosting Machine Training and Evaluation Framework

This package provides a complete framework for training, tuning, evaluating,
and interpreting Explainable Boosting Machine (EBM) models with automated
HTML report generation.

Main Components:
- EBMRunner: Main orchestrator for model training and evaluation
- EBMRunArtifacts: Container for run results and artifacts
- ValidationResult: Container for validation metrics

Example:
    >>> from ebm_runner import EBMRunner
    >>> from sklearn.datasets import load_breast_cancer
    >>> 
    >>> data = load_breast_cancer(as_frame=True)
    >>> X, y = data.data, data.target
    >>> 
    >>> runner = EBMRunner(output_dir="./my_ebm_run")
    >>> artifacts = runner.fit_optimize_validate(X, y)
    >>> print(f"Model saved to: {artifacts.model_path}")
    >>> print(f"Report saved to: {artifacts.report_html_path}")
"""

__version__ = "1.0.0"

from .runner import EBMRunner, EBMRunArtifacts
from .metrics import ValidationResult
from .config import RunConfig, PlotConfig, ReportConfig
from .feature_selection import FeatureSelectionResult

__all__ = [
    "EBMRunner",
    "EBMRunArtifacts",
    "ValidationResult",
    "FeatureSelectionResult",
    "RunConfig",
    "PlotConfig",
    "ReportConfig",
]
