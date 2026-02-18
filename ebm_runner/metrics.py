"""
Metrics calculation for classification and regression tasks.

This module provides functions to compute performance metrics for both
classification and regression models.
"""

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


@dataclass
class ValidationResult:
    """Container for validation results."""
    
    metrics: Dict[str, float]
    extra: Dict[str, Any]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        logger: Optional logger for warnings
        
    Returns:
        Dictionary of metric names and values
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    out: Dict[str, float] = {}
    
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute accuracy: {e}")
    
    try:
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    except Exception as e:
        logger.warning(f"Failed to compute F1 score: {e}")

    # AUC/AP if probabilities are available
    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            # Binary: shape (n,) or (n,2). Multiclass: (n,k)
            if y_proba.ndim == 1:
                proba_pos = y_proba
                out["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                out["avg_precision"] = float(average_precision_score(y_true, proba_pos))
                out["brier"] = float(brier_score_loss(y_true, proba_pos))
            else:
                if y_proba.shape[1] == 2:
                    proba_pos = y_proba[:, 1]
                    out["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                    out["avg_precision"] = float(average_precision_score(y_true, proba_pos))
                    out["brier"] = float(brier_score_loss(y_true, proba_pos))
                else:
                    # multiclass
                    out["roc_auc_ovr_macro"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    )
        except Exception as e:
            logger.warning(f"Failed to compute probability-based metrics: {e}")
    
    return out


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        logger: Optional logger for warnings
        
    Returns:
        Dictionary of metric names and values
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    out: Dict[str, float] = {}
    
    try:
        mse = mean_squared_error(y_true, y_pred)
        out["rmse"] = float(math.sqrt(mse))
    except Exception as e:
        logger.warning(f"Failed to compute RMSE: {e}")
    
    try:
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute MAE: {e}")
    
    try:
        out["r2"] = float(r2_score(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute R2: {e}")
    
    return out
