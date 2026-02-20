"""
Plotting functions for EBM model interpretation and performance visualization.

This module provides functions to create various plots for model interpretation
(global and local explanations) and performance evaluation (classification curves,
regression diagnostics).
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


def plot_feature_importances(
    global_exp,
    top_n: int = 20,
    figsize: tuple = (8, 6),
    logger: Optional[logging.Logger] = None
) -> plt.Figure:
    """
    Plot global feature importance scores from EBM model.
    
    Args:
        global_exp: Global explanation object from model.explain_global()
        top_n: Number of top features to display
        figsize: Figure size as (width, height)
        logger: Optional logger for warnings
        
    Returns:
        Matplotlib figure object
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    try:
        data = global_exp.data()
        names = data.get("names", [])
        scores = data.get("scores", [])

        if names is None or scores is None:
            names, scores = [], []

        # scores might include interactions at the end; we keep as-is but show top by abs score
        score_arr = np.asarray(scores, dtype=float)
        if score_arr.size == 0:
            fig = plt.figure(figsize=figsize)
            plt.title("Feature importances (no data)")
            return fig

        order = np.argsort(np.abs(score_arr))[::-1][:top_n]
        sel_names = [str(names[i]) for i in order]
        sel_scores = score_arr[order]

        fig = plt.figure(figsize=(figsize[0], max(3.5, 0.25 * len(order) + 1.5)))
        plt.barh(range(len(order))[::-1], sel_scores)
        plt.yticks(range(len(order))[::-1], sel_names)
        plt.xlabel("Importance score")
        plt.title(f"Top {len(order)} Feature Importances (EBM)")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot feature importances: {e}")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, f"Error plotting feature importances:\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")
        return fig


def plot_shape_for_feature(
    global_exp,
    feature_name: str,
    figsize: tuple = (7.5, 3.5),
    logger: Optional[logging.Logger] = None
) -> Optional[plt.Figure]:
    """
    Plot the learned shape function for one feature from global explanation.
    
    Args:
        global_exp: Global explanation object from model.explain_global()
        feature_name: Name of feature to plot
        figsize: Figure size as (width, height)
        logger: Optional logger for warnings
        
    Returns:
        Matplotlib figure object, or None if feature not found
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    try:
        data = global_exp.data()
        names = data.get("names", [])
        if feature_name not in names:
            logger.warning(f"Feature '{feature_name}' not found in global explanation")
            return None
        idx = list(names).index(feature_name)

        # Fetch detailed shape data for this feature
        detail_data = global_exp.data(idx)
        feature_values = detail_data.get("names", None) # In indexed data, 'names' holds the x-axis values
        feature_scores = detail_data.get("scores", None)

        if feature_values is None or feature_scores is None:
            # Try alternative keys
            feature_values = detail_data.get("values", None)
            
        if feature_values is None or feature_scores is None:
            logger.warning(f"No shape data available for feature '{feature_name}' (idx {idx})")
            return None

        fig = plt.figure(figsize=figsize)
        plt.title(f"Shape function: {feature_name}")

        # Categorical: v is list of category strings, s is list/array of contributions
        # Continuous: v often represents bin edges or bin midpoints
        if isinstance(feature_values, (list, tuple, np.ndarray)) and len(feature_values) > 0:
            if isinstance(feature_values[0], (str, bytes)):
                # Categorical
                x = np.arange(len(feature_values))
                plt.bar(x, np.asarray(feature_scores, dtype=float))
                plt.xticks(x, [str(xi) for xi in feature_values], rotation=45, ha="right")
                plt.ylabel("Contribution")
            else:
                # Continuous-ish
                vv = np.asarray(feature_values)
                ss = np.asarray(feature_scores, dtype=float)

                # If vv looks like edges (len = len(ss)+1), convert to midpoints
                if vv.ndim == 1 and ss.ndim == 1 and len(vv) == len(ss) + 1:
                    mids = 0.5 * (vv[:-1] + vv[1:])
                    plt.plot(mids, ss, marker="o", linewidth=1)
                    plt.xlabel("Feature value (bin midpoint)")
                else:
                    plt.plot(vv, ss, marker="o", linewidth=1)
                    plt.xlabel("Feature value")
                plt.ylabel("Contribution")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot shape for feature '{feature_name}': {e}")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, f"Error plotting shape:\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")
        return fig


def plot_local_explanation_bar(
    local_exp,
    sample_index: int = 0,
    top_n: int = 15,
    figsize: tuple = (8, 4),
    logger: Optional[logging.Logger] = None
) -> plt.Figure:
    """
    Plot local explanation for a single sample.
    
    Args:
        local_exp: Local explanation object from model.explain_local()
        sample_index: Index of sample to plot
        top_n: Number of top contributing features to show
        figsize: Figure size as (width, height)
        logger: Optional logger for warnings
        
    Returns:
        Matplotlib figure object
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    fig = plt.figure(figsize=figsize)
    plt.title(f"Local explanation (sample {sample_index})")

    try:
        # Fetch detailed data for this sample
        detail_data = local_exp.data(sample_index)
        names = detail_data.get("names", None)
        contrib = detail_data.get("scores", None)

        if names is None or contrib is None:
            plt.text(0.01, 0.5, "Local explanation data unavailable.", 
                    transform=plt.gca().transAxes)
            plt.axis("off")
            return fig

        contrib = np.asarray(contrib, dtype=float)
        if contrib.size == 0:
            plt.text(0.01, 0.5, "Empty contribution vector.", 
                    transform=plt.gca().transAxes)
            plt.axis("off")
            return fig

        order = np.argsort(np.abs(contrib))[::-1][:top_n]
        sel_names = [str(names[i]) for i in order]
        sel_scores = contrib[order]

        plt.barh(range(len(order))[::-1], sel_scores)
        plt.yticks(range(len(order))[::-1], sel_names)
        plt.xlabel("Contribution to prediction")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot local explanation: {e}")
        plt.text(0.5, 0.5, f"Error plotting local explanation:\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")
        return fig


def plot_classification_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: tuple = (5.5, 4),
    logger: Optional[logging.Logger] = None
) -> List[plt.Figure]:
    """
    Plot ROC, Precision-Recall, and Calibration curves for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        figsize: Figure size for each plot
        logger: Optional logger for warnings
        
    Returns:
        List of matplotlib figure objects
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    figs: List[plt.Figure] = []
    
    try:
        # Only for binary. For multiclass, you can extend with one-vs-rest plots.
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        elif y_proba.ndim == 1:
            y_score = y_proba
        else:
            logger.warning("Classification curves only supported for binary classification")
            return figs

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig1 = plt.figure(figsize=figsize)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        figs.append(fig1)

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fig2 = plt.figure(figsize=figsize)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        figs.append(fig2)

        # Calibration curve
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        fig3 = plt.figure(figsize=figsize)
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration curve")
        figs.append(fig3)
    
    except Exception as e:
        logger.error(f"Failed to plot classification curves: {e}")
    
    return figs


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (5.5, 4),
    logger: Optional[logging.Logger] = None
) -> List[plt.Figure]:
    """
    Plot regression diagnostic plots (residuals, distribution, predicted vs true).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size for each plot
        logger: Optional logger for warnings
        
    Returns:
        List of matplotlib figure objects
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    figs: List[plt.Figure] = []

    try:
        residuals = y_true - y_pred

        # Residuals vs predicted
        fig1 = plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, s=10, alpha=0.7)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual (true - pred)")
        plt.title("Residuals vs predicted")
        figs.append(fig1)

        # Residual distribution
        fig2 = plt.figure(figsize=figsize)
        plt.hist(residuals, bins=30)
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title("Residual distribution")
        figs.append(fig2)

        # Predicted vs true
        fig3 = plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, s=10, alpha=0.7)
        mn = min(np.min(y_true), np.min(y_pred))
        mx = max(np.max(y_true), np.max(y_pred))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Predicted vs true")
        figs.append(fig3)
    
    except Exception as e:
        logger.error(f"Failed to plot regression diagnostics: {e}")
    
    return figs


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (5.5, 4),
    logger: Optional[logging.Logger] = None
) -> plt.Figure:
    """
    Plot confusion matrix for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        figsize: Figure size
        logger: Optional logger for warnings
        
    Returns:
        Matplotlib figure object
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    try:
        fig = plt.figure(figsize=figsize)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(values_format="d", ax=plt.gca())
        plt.title("Confusion matrix")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, f"Error plotting confusion matrix:\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")
        return fig


def plot_interaction_shape(
    global_exp,
    interaction_name: str,
    figsize: tuple = (8, 6),
    logger: Optional[logging.Logger] = None
) -> Optional[plt.Figure]:
    """
    Plot the learned shape function for a feature interaction.
    
    Args:
        global_exp: Global explanation object from model.explain_global()
        interaction_name: Name of interaction (e.g., "feature1 & feature2")
        figsize: Figure size as (width, height)
        logger: Optional logger for warnings
        
    Returns:
        Matplotlib figure object, or None if interaction not found
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    try:
        data = global_exp.data()
        names = data.get("names", [])
        
        if interaction_name not in names:
            logger.warning(f"Interaction '{interaction_name}' not found in global explanation")
            return None
        
        idx = list(names).index(interaction_name)
        
        # Fetch detailed interaction data
        detail_data = global_exp.data(idx)
        # For interactions, names often holds bin info, scores holds the matrix
        interaction_scores = detail_data.get("scores", None)
        
        if interaction_scores is None:
            logger.warning(f"No score data available for interaction '{interaction_name}'")
            return None
        
        fig = plt.figure(figsize=figsize)
        plt.title(f"Interaction: {interaction_name}")
        
        # For 2D interactions, create heatmap
        if isinstance(interaction_scores, np.ndarray) and interaction_scores.ndim == 2:
            im = plt.imshow(interaction_scores, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, label='Contribution')
            plt.xlabel("Feature 2 bins")
            plt.ylabel("Feature 1 bins")
        else:
            # Fallback to 1D plot
            scores_flat = np.asarray(interaction_scores, dtype=float).flatten()
            plt.plot(scores_flat, marker='o', linewidth=1)
            plt.xlabel("Interaction bin")
            plt.ylabel("Contribution")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot interaction '{interaction_name}': {e}")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, f"Error plotting interaction:\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")
        return fig


def fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """
    Convert matplotlib figure to base64-encoded PNG string for HTML embedding.
    
    Args:
        fig: Matplotlib figure object
        dpi: Resolution for the output image
        
    Returns:
        Base64-encoded PNG image string
    """
    import io
    import base64
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_base64


def plot_elimination_curve(
    history,
    tolerance: float = 0.02,
    figsize: tuple = (9, 4.5),
    logger: Optional[logging.Logger] = None,
) -> plt.Figure:
    """
    Plot the feature-elimination curve.

    Shows the CV score at each elimination step with a shaded tolerance
    band around the baseline.

    Args:
        history: DataFrame with columns ``n_features`` and ``cv_score``
            (as produced by ``FeatureSelectionResult.history``).
        tolerance: The tolerance that was used during elimination.
        figsize: Figure size.
        logger: Optional logger.

    Returns:
        Matplotlib figure object.
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")

    fig, ax = plt.subplots(figsize=figsize)

    n_feats = history["n_features"].values
    scores = history["cv_score"].values
    baseline = scores[0]

    ax.plot(n_feats, scores, marker="o", linewidth=2, color="#2980b9", zorder=3)

    # Tolerance band
    abs_baseline = abs(baseline) if abs(baseline) > 1e-12 else 1e-12
    lower_bound = baseline - tolerance * abs_baseline
    ax.axhline(baseline, linestyle="--", color="#27ae60", linewidth=1, label="Baseline")
    ax.axhline(lower_bound, linestyle="--", color="#e74c3c", linewidth=1, label="Tolerance limit")
    ax.fill_between(
        [n_feats.min() - 0.5, n_feats.max() + 0.5],
        lower_bound, baseline,
        alpha=0.10, color="#27ae60",
    )

    # Annotate removed features
    removed = history["removed_feature"].values
    for i in range(1, len(n_feats)):
        if removed[i] is not None and str(removed[i]) != "None":
            ax.annotate(
                str(removed[i]),
                xy=(n_feats[i], scores[i]),
                xytext=(0, -18),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                rotation=30,
                color="#555",
            )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("CV score")
    ax.set_title("Feature Elimination Curve")
    ax.legend(loc="best", fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    return fig


def plot_feature_density(
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    feature_name: str,
    is_classification: bool,
    figsize: tuple = (5.5, 4),
    logger: Optional[logging.Logger] = None
) -> Optional[plt.Figure]:
    """
    Plot feature density (KDE), potentially split by class or target value.
    
    Args:
        X_df: Feature DataFrame
        y_arr: Target array
        feature_name: Feature to plot
        is_classification: Whether this is a classification task
        figsize: Figure size
        logger: Optional logger
        
    Returns:
        Matplotlib figure or None
    """
    if logger is None:
        logger = logging.getLogger("ebm_runner")
    
    if feature_name not in X_df.columns:
        logger.warning(f"Feature '{feature_name}' not in X_df")
        return None

    try:
        series = X_df[feature_name]
        
        # Skip if not numeric
        if not np.issubdtype(series.dtype, np.number):
            return None

        fig = plt.figure(figsize=figsize)
        
        if is_classification:
            # Split by class
            classes = np.unique(y_arr)
            for cls in classes:
                mask = (y_arr == cls)
                subset = series[mask].dropna()
                if not subset.empty:
                    subset.plot.kde(label=f"Class {cls}", ax=plt.gca())
            plt.title(f"Density: {feature_name} (by Class)")
        else:
            # Regression: Split by median to show distribution differences for high/low target
            median_y = np.median(y_arr)
            low_mask = (y_arr <= median_y)
            high_mask = (y_arr > median_y)
            
            low_subset = series[low_mask].dropna()
            high_subset = series[high_mask].dropna()
            
            if not low_subset.empty:
                low_subset.plot.kde(label="Low Target", ax=plt.gca())
            if not high_subset.empty:
                high_subset.plot.kde(label="High Target", ax=plt.gca())
                
            plt.title(f"Density: {feature_name} (by Target Median)")

        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return fig

    except Exception as e:
        logger.error(f"Failed to plot density for {feature_name}: {e}")
        return None


