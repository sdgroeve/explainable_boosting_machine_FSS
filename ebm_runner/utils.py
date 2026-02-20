"""
Utility functions for EBM Runner.

This module contains general utility functions for logging, file operations,
data conversion, and task detection.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_logger(name: str = "ebm_runner", level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_fig_png(fig: plt.Figure, out_path: str, dpi: int = 200) -> str:
    """
    Save matplotlib figure as PNG file.
    
    Args:
        fig: Matplotlib figure to save
        out_path: Output file path
        dpi: Dots per inch for output image
        
    Returns:
        Path to saved file
    """
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def safe_json(obj: Any) -> str:
    """
    Safely convert object to JSON string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception as e:
        logging.getLogger("ebm_runner").warning(f"JSON serialization failed: {e}")
        return str(obj)


def df_to_table_data(df: pd.DataFrame, max_rows: int = 20) -> List[List[Any]]:
    """
    Convert DataFrame to table data format for ReportLab.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to include
        
    Returns:
        List of lists with header row and data rows
    """
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    df2 = df2.reset_index(drop=True)
    header = list(df2.columns)
    rows = df2.values.tolist()
    return [header] + rows


def wrap_long(s: str, n: int = 110) -> str:
    """
    Wrap long strings with HTML line breaks.
    
    Args:
        s: String to wrap
        n: Maximum length before wrapping
        
    Returns:
        Wrapped string with <br/> tags
    """
    s = str(s)
    if len(s) <= n:
        return s
    chunks = [s[i:i+n] for i in range(0, len(s), n)]
    return "<br/>".join(chunks)


def sanitize_filename(s: str, max_length: int = 120) -> str:
    """
    Sanitize string for use as filename.
    
    Args:
        s: String to sanitize
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename string
    """
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:max_length]


def is_classification_target(
    y: Union[pd.Series, np.ndarray], 
    force: Optional[str] = None
) -> bool:
    """
    Detect whether target variable is for classification or regression.
    
    Args:
        y: Target variable
        force: Force task type ('classification', 'clf', 'class', 'regression', 'reg', 'regr')
        
    Returns:
        True if classification, False if regression
        
    Raises:
        ValueError: If force parameter is invalid
    """
    if force is not None:
        force = force.lower().strip()
        if force in {"classification", "clf", "class"}:
            return True
        if force in {"regression", "reg", "regr"}:
            return False
        raise ValueError(
            f"force_task must be one of: classification/clf/class or regression/reg/regr, got '{force}'"
        )

    y_series = pd.Series(y).dropna()
    
    # If it's objects/strings/categories, it's definitely classification
    if y_series.dtype == 'object' or isinstance(y_series.dtype, pd.CategoricalDtype) or y_series.dtype == 'bool':
        return True

    # Heuristic for numeric types: small number of unique values implies classes
    unique = y_series.unique()
    if len(unique) <= 20:
        return True
        
    return False


def validate_input_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Validate input data for common issues.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Optional feature names
        logger: Optional logger for warnings
        
    Raises:
        ValueError: If data validation fails
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    # Check empty data
    if len(X) == 0:
        raise ValueError("Input data X is empty")
    if len(y) == 0:
        raise ValueError("Target variable y is empty")
    
    # Check shape compatibility
    if len(X) != len(y):
        raise ValueError(f"X and y have incompatible lengths: {len(X)} vs {len(y)}")
    
    # Check for NaN/inf in target
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.floating) or np.issubdtype(y_arr.dtype, np.integer):
        if np.any(np.isnan(y_arr.astype(float))):
            raise ValueError("Target variable y contains NaN values")
        if np.any(np.isinf(y_arr.astype(float))):
            raise ValueError("Target variable y contains infinite values")
    else:
        # For non-numeric targets (e.g. string labels), check for None / empty
        y_series = pd.Series(y)
        if y_series.isna().any():
            raise ValueError("Target variable y contains missing (NaN/None) values")
    
    # Check feature names
    if isinstance(X, pd.DataFrame):
        n_features = X.shape[1]
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) doesn't match "
                f"number of features in X ({n_features})"
            )
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        n_features = X.shape[1]
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) doesn't match "
                f"number of features in X ({n_features})"
            )
    
    # Warn about NaN in features
    if isinstance(X, pd.DataFrame):
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Features contain NaN values: {nan_cols[:5]}")
    elif isinstance(X, np.ndarray):
        if np.any(np.isnan(X)):
            logger.warning("Feature matrix X contains NaN values")
