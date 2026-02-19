"""
HTML report generation for EBM model results.

This module provides the HTMLReportGenerator class for creating comprehensive
HTML reports with model performance metrics, interpretability plots, and
feature interaction analysis.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ReportConfig
from .metrics import compute_classification_metrics, compute_regression_metrics
from .plotting import (
    plot_feature_importances,
    plot_shape_for_feature,
    plot_local_explanation_bar,
    plot_classification_curves,
    plot_regression_diagnostics,
    plot_confusion_matrix,
    plot_interaction_shape,
    plot_elimination_curve,
    fig_to_base64,
)
from .html_templates import (
    HTML_TEMPLATE,
    TITLE_SECTION,
    SUMMARY_SECTION,
    TOC_SECTION,
    TOC_SECTION_WITH_FS,
    TOC_SECTION_WITH_BOTH_FS,
    TOC_SECTION_WITH_FWD_FS,
    PERFORMANCE_SECTION_START,
    METRICS_TABLE_START,
    METRICS_TABLE_ROW,
    METRICS_TABLE_END,
    PLOTS_SECTION_START,
    PLOT_ITEM,
    PLOTS_SECTION_END,
    GLOBAL_IMPORTANCE_SECTION,
    INDIVIDUAL_FEATURES_SECTION_START,
    FEATURE_SHAPE_SECTION,
    INTERACTIONS_SECTION_START,
    INTERACTION_ITEM,
    LOCAL_SECTION_START,
    LOCAL_EXPLANATION_ITEM,
    NO_DATA_MESSAGE,
    FEATURE_SELECTION_SECTION_START,
    FEATURE_SELECTION_SUMMARY,
    FEATURE_SELECTION_SELECTED_LIST,
    FEATURE_SELECTION_HISTORY_TABLE_START,
    FEATURE_SELECTION_HISTORY_ROW,
    FEATURE_SELECTION_HISTORY_TABLE_END,
    FEATURE_SELECTION_PLOT,
    FORWARD_SELECTION_SECTION_START,
    EVAL_NOTE_TRAIN_TEST,
    EVAL_NOTE_CV_ONLY,
    EVAL_NOTE_TRAIN_ONLY,
    TUNING_SECTION_START,
    TUNING_TABLE_START,
    TUNING_TABLE_ROW,
    TUNING_TABLE_END,
)


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for EBM model runs."""
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HTML report generator.
        
        Args:
            config: Report configuration
            logger: Optional logger
        """
        self.config = config or ReportConfig()
        self.logger = logger or logging.getLogger("ebm_runner")
    
    def generate_report(
        self,
        html_path: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        best_model: Any,
        is_classification: bool,
        best_params: Dict[str, Any],
        cv_results: pd.DataFrame,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        feature_selection_result: Optional[Any] = None,
        forward_selection_result: Optional[Any] = None,
        eval_strategy: str = "train_test",
    ) -> None:
        """
        Generate comprehensive HTML report.
        
        Args:
            html_path: Output HTML file path
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            best_model: Trained EBM model
            is_classification: Whether task is classification
            best_params: Best hyperparameters from tuning
            cv_results: Cross-validation results DataFrame
            y_pred: Predictions on test set
            y_proba: Predicted probabilities (classification only)
            feature_selection_result: Optional FeatureSelectionResult (backward)
            forward_selection_result: Optional FeatureSelectionResult (forward)
        """
        self.logger.info("Generating HTML report...")
        
        content_parts = []
        
        # Build report sections
        content_parts.append(self._add_title())
        content_parts.append(self._add_summary(
            X_train, X_test, is_classification, eval_strategy=eval_strategy,
        ))
        # Pick the right TOC variant
        has_bw = feature_selection_result is not None
        has_fw = forward_selection_result is not None
        if has_bw and has_fw:
            content_parts.append(TOC_SECTION_WITH_BOTH_FS)
        elif has_bw:
            content_parts.append(TOC_SECTION_WITH_FS)
        elif has_fw:
            content_parts.append(TOC_SECTION_WITH_FWD_FS)
        else:
            content_parts.append(TOC_SECTION)

        content_parts.append(self._add_performance_section(
            y_test, y_pred, y_proba, is_classification,
            eval_strategy=eval_strategy,
        ))
        content_parts.append(self._add_global_interpretation(best_model))
        content_parts.append(self._add_individual_features(best_model))
        content_parts.append(self._add_interactions(best_model))
        content_parts.append(self._add_local_explanations(best_model, X_test, y_test))

        # Hyperparameter tuning (section 6)
        content_parts.append(self._add_tuning_section(cv_results, section_number=6))

        # Backward feature selection (section 7)
        if feature_selection_result:
            content_parts.append(self._add_feature_selection(
                feature_selection_result,
                section_start_template=FEATURE_SELECTION_SECTION_START,
                section_number=7,
            ))

        # Forward feature selection / redundancy analysis (section 7 or 8)
        if forward_selection_result:
            fwd_section_num = 8 if feature_selection_result else 7
            content_parts.append(self._add_feature_selection(
                forward_selection_result,
                section_start_template=FORWARD_SELECTION_SECTION_START,
                section_number=fwd_section_num,
            ))
        
        # Combine all content
        full_content = "\n".join(content_parts)
        html_output = HTML_TEMPLATE.format(content=full_content)
        
        # Write to file
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            self.logger.info(f"HTML report saved: {html_path}")
        except Exception as e:
            self.logger.error(f"Failed to write HTML report: {e}")
            raise
    
    def _add_title(self) -> str:
        """Add report title with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return TITLE_SECTION.format(timestamp=timestamp)
    
    def _add_summary(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        is_classification: bool,
        *,
        eval_strategy: str = "train_test",
    ) -> str:
        """Add run summary section."""
        strategy_labels = {
            "train_test": "Train / Test Split",
            "cv_only": "Cross-Validation Only",
            "train_only": "Train Only (no holdout)",
        }
        strategy_notes = {
            "train_test": EVAL_NOTE_TRAIN_TEST,
            "cv_only": EVAL_NOTE_CV_ONLY,
            "train_only": EVAL_NOTE_TRAIN_ONLY,
        }
        return SUMMARY_SECTION.format(
            task_type="Classification" if is_classification else "Regression",
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=X_train.shape[1],
            eval_strategy_label=strategy_labels.get(eval_strategy, eval_strategy),
            eval_strategy_note=strategy_notes.get(eval_strategy, ""),
        )
    
    def _add_performance_section(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        is_classification: bool,
        *,
        eval_strategy: str = "train_test",
    ) -> str:
        """Add performance metrics and validation plots."""
        descriptions = {
            "train_test": "Comprehensive evaluation of model performance on the holdout test set.",
            "cv_only": "Performance metrics aggregated from out-of-fold cross-validation predictions.",
            "train_only": "Performance metrics computed on the training data (may overestimate generalisation).",
        }
        parts = [PERFORMANCE_SECTION_START.format(
            performance_description=descriptions.get(eval_strategy, descriptions["train_test"]),
        )]
        
        # Compute metrics
        if is_classification:
            metrics = compute_classification_metrics(y_test, y_pred, y_proba, self.logger)
        else:
            metrics = compute_regression_metrics(y_test, y_pred, self.logger)
        
        # Add metrics table
        parts.append(METRICS_TABLE_START)
        for metric_name, metric_value in metrics.items():
            parts.append(METRICS_TABLE_ROW.format(
                metric_name=metric_name,
                metric_value=f"{metric_value:.6f}"
            ))
        parts.append(METRICS_TABLE_END)
        
        # Add validation plots
        parts.append(PLOTS_SECTION_START)
        
        if is_classification:
            # Confusion matrix
            fig = plot_confusion_matrix(y_test, y_pred, logger=self.logger)
            img_b64 = fig_to_base64(fig)
            parts.append(PLOT_ITEM.format(
                img_base64=img_b64,
                alt_text="Confusion Matrix"
            ))
            
            # Classification curves (binary only)
            if y_proba is not None:
                for i, fig in enumerate(
                    plot_classification_curves(y_test, y_proba, logger=self.logger),
                    start=1
                ):
                    img_b64 = fig_to_base64(fig)
                    curve_names = ["ROC Curve", "Precision-Recall Curve", "Calibration Curve"]
                    alt_text = curve_names[i-1] if i <= len(curve_names) else f"Curve {i}"
                    parts.append(PLOT_ITEM.format(
                        img_base64=img_b64,
                        alt_text=alt_text
                    ))
        else:
            # Regression diagnostics
            for i, fig in enumerate(
                plot_regression_diagnostics(y_test, y_pred, logger=self.logger),
                start=1
            ):
                img_b64 = fig_to_base64(fig)
                diag_names = ["Residuals vs Predicted", "Residual Distribution", "Predicted vs True"]
                alt_text = diag_names[i-1] if i <= len(diag_names) else f"Diagnostic {i}"
                parts.append(PLOT_ITEM.format(
                    img_base64=img_b64,
                    alt_text=alt_text
                ))
        
        parts.append(PLOTS_SECTION_END)
        
        return "\n".join(parts)
    
    def _add_global_interpretation(self, best_model: Any) -> str:
        """Add global feature importance section."""
        try:
            global_exp = best_model.explain_global()
            
            # Plot all features (not limited to top N)
            data = global_exp.data()
            names = data.get("names", [])
            scores = data.get("scores", [])
            
            # Count only individual features (not interactions)
            n_features = sum(1 for name in names if not self._is_interaction(name))
            
            fig = plot_feature_importances(global_exp, top_n=n_features, logger=self.logger)
            img_b64 = fig_to_base64(fig)
            
            return GLOBAL_IMPORTANCE_SECTION.format(importance_plot=img_b64)
        
        except Exception as e:
            self.logger.error(f"Failed to generate global interpretation: {e}")
            return f"<h2 id='global-importance'>2. Global Feature Importance</h2>{NO_DATA_MESSAGE}"
    
    def _add_individual_features(self, best_model: Any) -> str:
        """Add individual feature analysis with shape plots for ALL features."""
        parts = [INDIVIDUAL_FEATURES_SECTION_START]
        
        try:
            global_exp = best_model.explain_global()
            data = global_exp.data()
            names = data.get("names", [])
            scores = data.get("scores", [])
            
            # Get all individual features (not interactions) with their scores
            features_with_scores: List[Tuple[str, float]] = []
            for i, name in enumerate(names):
                if not self._is_interaction(name):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    features_with_scores.append((str(name), score))
            
            # Sort by absolute importance
            features_with_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            self.logger.info(f"Generating shape plots for {len(features_with_scores)} features...")
            
            # Generate shape plot for each feature
            for feature_name, importance_score in features_with_scores:
                fig = plot_shape_for_feature(global_exp, feature_name, logger=self.logger)
                if fig is not None:
                    img_b64 = fig_to_base64(fig)
                    parts.append(FEATURE_SHAPE_SECTION.format(
                        feature_name=feature_name,
                        importance_score=f"{importance_score:.6f}",
                        shape_plot=img_b64
                    ))
        
        except Exception as e:
            self.logger.error(f"Failed to generate individual feature analysis: {e}")
            parts.append(NO_DATA_MESSAGE)
        
        return "\n".join(parts)
    
    def _add_interactions(self, best_model: Any) -> str:
        """Add feature interaction analysis."""
        parts = [INTERACTIONS_SECTION_START]
        
        try:
            global_exp = best_model.explain_global()
            data = global_exp.data()
            names = data.get("names", [])
            scores = data.get("scores", [])
            
            # Get all interactions with their scores
            interactions_with_scores: List[Tuple[str, float]] = []
            for i, name in enumerate(names):
                if self._is_interaction(name):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    interactions_with_scores.append((str(name), score))
            
            if not interactions_with_scores:
                parts.append("<p>No feature interactions detected in this model.</p>")
                return "\n".join(parts)
            
            # Sort by absolute importance
            interactions_with_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            self.logger.info(f"Generating plots for {len(interactions_with_scores)} interactions...")
            
            # Generate plot for each interaction
            for interaction_name, interaction_score in interactions_with_scores:
                fig = plot_interaction_shape(global_exp, interaction_name, logger=self.logger)
                if fig is not None:
                    img_b64 = fig_to_base64(fig)
                    parts.append(INTERACTION_ITEM.format(
                        interaction_name=interaction_name,
                        interaction_score=f"{interaction_score:.6f}",
                        interaction_plot=img_b64
                    ))
        
        except Exception as e:
            self.logger.error(f"Failed to generate interaction analysis: {e}")
            parts.append(NO_DATA_MESSAGE)
        
        return "\n".join(parts)
    
    def _add_local_explanations(
        self,
        best_model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> str:
        """Add local explanation section."""
        parts = [LOCAL_SECTION_START]
        
        n_local = min(self.config.n_local_samples, len(X_test))
        
        if n_local > 0:
            try:
                local_exp = best_model.explain_local(X_test.iloc[:n_local], y_test[:n_local])
                
                for i in range(n_local):
                    fig = plot_local_explanation_bar(local_exp, sample_index=i, top_n=15, logger=self.logger)
                    img_b64 = fig_to_base64(fig)
                    
                    parts.append(LOCAL_EXPLANATION_ITEM.format(
                        sample_idx=i,
                        true_value=f"{y_test[i]:.4f}" if isinstance(y_test[i], (int, float)) else str(y_test[i]),
                        pred_value=f"{best_model.predict(X_test.iloc[[i]])[0]:.4f}",
                        local_plot=img_b64
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to generate local explanations: {e}")
                parts.append(NO_DATA_MESSAGE)
        else:
            parts.append("<p>No local explanations generated.</p>")
        
        return "\n".join(parts)
    
    def _add_feature_selection(
        self,
        fs_result: Any,
        section_start_template: str = FEATURE_SELECTION_SECTION_START,
        section_number: int = 6,
    ) -> str:
        """Add feature selection results section."""
        parts = [section_start_template.format(section_number=section_number)]

        try:
            n_original = fs_result.history["n_features"].iloc[0]
            n_selected = len(fs_result.selected_features)
            abs_base = abs(fs_result.baseline_score) if abs(fs_result.baseline_score) > 1e-12 else 1e-12
            rel_drop = (fs_result.baseline_score - fs_result.final_score) / abs_base
            tolerance = fs_result.tolerance

            parts.append(FEATURE_SELECTION_SUMMARY.format(
                n_original=n_original,
                n_selected=n_selected,
                tolerance=f"{tolerance:.4f} ({tolerance * 100:.1f}%)",
                baseline_score=f"{fs_result.baseline_score:.6f}",
                final_score=f"{fs_result.final_score:.6f}",
                relative_drop=f"{rel_drop:.4f} ({rel_drop * 100:.2f}%)",
            ))

            # Selected features list
            badges = " &nbsp; ".join(
                f"<code>{f}</code>" for f in fs_result.selected_features
            )
            parts.append(FEATURE_SELECTION_SELECTED_LIST.format(feature_badges=badges))

            # Elimination history table
            parts.append(FEATURE_SELECTION_HISTORY_TABLE_START)
            for _, row in fs_result.history.iterrows():
                parts.append(FEATURE_SELECTION_HISTORY_ROW.format(
                    step=int(row["step"]),
                    n_features=int(row["n_features"]),
                    removed_feature=row["removed_feature"] if row["removed_feature"] is not None else "—",
                    cv_score=f"{row['cv_score']:.6f}",
                    delta=f"{row['delta_from_baseline']:+.6f}",
                    relative_drop=f"{row['relative_drop']:.4f}",
                ))
            parts.append(FEATURE_SELECTION_HISTORY_TABLE_END)

            # Elimination curve plot
            fig = plot_elimination_curve(
                fs_result.history,
                tolerance=tolerance,
                logger=self.logger,
            )
            img_b64 = fig_to_base64(fig)
            parts.append(FEATURE_SELECTION_PLOT.format(elimination_plot=img_b64))

        except Exception as e:
            self.logger.error(f"Failed to generate feature selection section: {e}")
            parts.append(NO_DATA_MESSAGE)

        return "\n".join(parts)

    def _add_tuning_section(
        self,
        cv_results: pd.DataFrame,
        section_number: int = 6,
    ) -> str:
        """Add hyperparameter tuning results section."""
        if cv_results is None or cv_results.empty:
            return f"<h2 id='hyperparameter-tuning'>{section_number}. Hyperparameter Tuning</h2>{NO_DATA_MESSAGE}"

        # Detect search method
        # Since we don't pass the search object, we guess from cv_results
        # RandomizedSearchCV usually has 'mean_fit_time' and 'std_fit_time'
        method = "RandomizedSearchCV" if "mean_fit_time" in cv_results.columns else "Hyperparameter Search"
        
        # Best params are in the first row after sorting by rank (which we expect cv_results to be)
        best_row = cv_results.iloc[0]
        param_cols = [c for c in cv_results.columns if c.startswith("param_") and not c.endswith("_")]
        best_params_dict = {c.replace("param_", ""): best_row[c] for c in param_cols}
        best_params_display = " &nbsp; ".join(f"<code>{k}={v}</code>" for k, v in best_params_dict.items())

        n_candidates = len(cv_results)
        # Find n_folds from split score columns
        split_cols = [c for c in cv_results.columns if c.startswith("split") and c.endswith("_test_score")]
        n_folds = len(split_cols)

        parts = [TUNING_SECTION_START.format(
            section_number=section_number,
            method=method,
            n_candidates=n_candidates,
            n_folds=n_folds,
            best_params_display=best_params_display
        )]

        parts.append(TUNING_TABLE_START)
        for _, row in cv_results.iterrows():
            params_dict = {c.replace("param_", ""): row[c] for c in param_cols}
            params_str = ", ".join(f"{k}={v}" for k, v in params_dict.items())
            parts.append(TUNING_TABLE_ROW.format(
                rank=int(row["rank_test_score"]),
                mean_score=float(row["mean_test_score"]),
                std_score=float(row["std_test_score"]),
                params=params_str
            ))
        parts.append(TUNING_TABLE_END)

        return "\n".join(parts)

    def _is_interaction(self, name: str) -> bool:
        """Check if a feature name represents an interaction."""
        name_str = str(name)
        return " & " in name_str or (name_str.startswith("(") and name_str.endswith(")"))
