"""
PDF report generation for EBM model results.

This module provides the PDFReportGenerator class for creating comprehensive
PDF reports with model performance metrics, interpretability plots, and
hyperparameter tuning results.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

from .config import ReportConfig
from .utils import save_fig_png, df_to_table_data, wrap_long, sanitize_filename
from .metrics import compute_classification_metrics, compute_regression_metrics
from .plotting import (
    plot_feature_importances,
    plot_shape_for_feature,
    plot_local_explanation_bar,
    plot_classification_curves,
    plot_regression_diagnostics,
    plot_confusion_matrix,
)


class PDFReportGenerator:
    """Generate comprehensive PDF reports for EBM model runs."""
    
    def __init__(
        self,
        plots_dir: str,
        config: Optional[ReportConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PDF report generator.
        
        Args:
            plots_dir: Directory to save plot images
            config: Report configuration
            logger: Optional logger
        """
        self.plots_dir = plots_dir
        self.config = config or ReportConfig()
        self.logger = logger or logging.getLogger("ebm_runner")
        
        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self.style_h = self.styles["Heading1"]
        self.style_h2 = self.styles["Heading2"]
        self.style_n = self.styles["BodyText"]
    
    def generate_report(
        self,
        pdf_path: str,
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
    ) -> None:
        """
        Generate comprehensive PDF report.
        
        Args:
            pdf_path: Output PDF file path
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
        """
        self.logger.info("Generating PDF report...")
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm,
            title="EBM Model Report",
            author="EBMRunner",
        )

        story: List[Any] = []

        # Build report sections
        story.extend(self._add_title())
        story.extend(self._add_summary_section(X_train, X_test, is_classification))
        story.extend(self._add_data_overview(X_train))
        story.extend(self._add_hyperparameters(best_params))
        story.extend(self._add_cv_results(cv_results))
        story.append(PageBreak())
        
        story.extend(self._add_performance_section(
            y_test, y_pred, y_proba, is_classification
        ))
        story.append(PageBreak())
        
        story.extend(self._add_global_interpretation(best_model))
        story.append(PageBreak())
        
        story.extend(self._add_local_interpretation(best_model, X_test, y_test))

        # Build PDF
        try:
            doc.build(story)
            self.logger.info(f"PDF report saved: {pdf_path}")
        except Exception as e:
            self.logger.error(f"Failed to build PDF: {e}")
            raise
    
    def _add_title(self) -> List[Any]:
        """Add report title."""
        return [
            Paragraph("Explainable Boosting Machine (EBM) - Model Report", self.style_h),
            Spacer(1, 8),
        ]
    
    def _add_summary_section(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        is_classification: bool
    ) -> List[Any]:
        """Add run summary section."""
        elements = [
            Paragraph("Run Summary", self.style_h2),
            Spacer(1, 6),
            Paragraph(
                f"Task: <b>{'Classification' if is_classification else 'Regression'}</b>",
                self.style_n
            ),
            Paragraph(
                f"Train rows: <b>{len(X_train)}</b> | Test rows: <b>{len(X_test)}</b>",
                self.style_n
            ),
            Paragraph(f"Features: <b>{X_train.shape[1]}</b>", self.style_n),
            Spacer(1, 10),
        ]
        return elements
    
    def _add_data_overview(self, X_train: pd.DataFrame) -> List[Any]:
        """Add data overview section."""
        elements = [
            Paragraph("Data Overview", self.style_h2),
            Spacer(1, 6),
        ]
        
        dtypes = pd.DataFrame({
            "feature": X_train.columns,
            "dtype": [str(t) for t in X_train.dtypes]
        })
        elements.extend(self._add_table(dtypes, max_rows=25))
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _add_hyperparameters(self, best_params: Dict[str, Any]) -> List[Any]:
        """Add best hyperparameters section."""
        elements = [
            Paragraph("Best Hyperparameters", self.style_h2),
            Spacer(1, 6),
        ]
        
        best_params_df = pd.DataFrame({
            "param": list(best_params.keys()),
            "value": [str(v) for v in best_params.values()]
        })
        elements.extend(self._add_table(best_params_df, max_rows=50))
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _add_cv_results(self, cv_results: pd.DataFrame) -> List[Any]:
        """Add cross-validation results section."""
        elements = [
            Paragraph("Cross-Validation Results (Top 20)", self.style_h2),
            Spacer(1, 6),
        ]
        
        cols = [
            c for c in cv_results.columns
            if c in {"rank_test_score", "mean_test_score", "std_test_score"}
            or c.startswith("param_")
        ]
        leaderboard = cv_results[cols].head(20).copy()
        elements.extend(self._add_table(leaderboard, max_rows=20))
        
        return elements
    
    def _add_performance_section(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        is_classification: bool
    ) -> List[Any]:
        """Add holdout performance section with metrics and plots."""
        elements = [
            Paragraph("Holdout Test Performance", self.style_h2),
            Spacer(1, 6),
        ]
        
        # Compute metrics
        if is_classification:
            metrics = compute_classification_metrics(y_test, y_pred, y_proba, self.logger)
        else:
            metrics = compute_regression_metrics(y_test, y_pred, self.logger)
        
        met_df = pd.DataFrame({
            "metric": list(metrics.keys()),
            "value": [f"{v:.6f}" for v in metrics.values()]
        })
        elements.extend(self._add_table(met_df, max_rows=50))
        elements.append(Spacer(1, 10))
        
        # Generate plots
        plot_paths: List[str] = []
        
        if is_classification:
            # Confusion matrix
            fig = plot_confusion_matrix(y_test, y_pred, logger=self.logger)
            p = os.path.join(self.plots_dir, "confusion_matrix.png")
            plot_paths.append(save_fig_png(fig, p))
            
            # Curves (binary only)
            if y_proba is not None:
                for i, fig in enumerate(
                    plot_classification_curves(y_test, y_proba, logger=self.logger),
                    start=1
                ):
                    p = os.path.join(self.plots_dir, f"class_curve_{i}.png")
                    plot_paths.append(save_fig_png(fig, p))
        else:
            for i, fig in enumerate(
                plot_regression_diagnostics(y_test, y_pred, logger=self.logger),
                start=1
            ):
                p = os.path.join(self.plots_dir, f"reg_diag_{i}.png")
                plot_paths.append(save_fig_png(fig, p))
        
        if plot_paths:
            elements.append(Paragraph("Validation Plots", self.style_h2))
            elements.append(Spacer(1, 6))
            elements.extend(self._add_images_grid(
                plot_paths,
                images_per_row=self.config.images_per_row,
                img_width_cm=self.config.image_width_cm
            ))
        
        return elements
    
    def _add_global_interpretation(self, best_model: Any) -> List[Any]:
        """Add global interpretation section."""
        elements = [
            Paragraph("Model Interpretation (Global)", self.style_h2),
            Spacer(1, 6),
        ]
        
        try:
            global_exp = best_model.explain_global()
            
            # Feature importances
            fig = plot_feature_importances(global_exp, top_n=25, logger=self.logger)
            fi_path = os.path.join(self.plots_dir, "feature_importances.png")
            save_fig_png(fig, fi_path)
            elements.extend(self._add_images_grid([fi_path], images_per_row=1, img_width_cm=16.5))
            elements.append(Spacer(1, 8))
            
            # Shape functions for top features
            top_feats = self._get_top_features(global_exp)
            shape_paths: List[str] = []
            
            for feat in top_feats:
                fig = plot_shape_for_feature(global_exp, feat, logger=self.logger)
                if fig is None:
                    continue
                p = os.path.join(self.plots_dir, f"shape_{sanitize_filename(feat)}.png")
                shape_paths.append(save_fig_png(fig, p))
            
            if shape_paths:
                elements.append(Paragraph(f"Shape Functions (Top {len(shape_paths)})", self.style_h2))
                elements.append(Spacer(1, 6))
                elements.extend(self._add_images_grid(shape_paths, images_per_row=1, img_width_cm=16.5))
            else:
                elements.append(
                    Paragraph("Shape functions could not be extracted for this model/version.", self.style_n)
                )
        
        except Exception as e:
            self.logger.error(f"Failed to generate global interpretation: {e}")
            elements.append(Paragraph(f"Error generating global interpretation: {str(e)}", self.style_n))
        
        return elements
    
    def _add_local_interpretation(
        self,
        best_model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> List[Any]:
        """Add local interpretation section."""
        elements = [
            Paragraph("Model Interpretation (Local)", self.style_h2),
            Spacer(1, 6),
            Paragraph(
                "Local explanations show per-feature contributions for individual predictions.",
                self.style_n,
            ),
            Spacer(1, 8),
        ]
        
        n_local = min(self.config.n_local_samples, len(X_test))
        local_paths: List[str] = []
        
        if n_local > 0:
            try:
                local_exp = best_model.explain_local(X_test.iloc[:n_local], y_test[:n_local])
                for i in range(n_local):
                    fig = plot_local_explanation_bar(local_exp, sample_index=i, top_n=15, logger=self.logger)
                    p = os.path.join(self.plots_dir, f"local_{i}.png")
                    local_paths.append(save_fig_png(fig, p))
            except Exception as e:
                self.logger.warning(f"Failed to generate local explanations: {e}")
                local_paths = []
        
        if local_paths:
            elements.extend(self._add_images_grid(local_paths, images_per_row=1, img_width_cm=16.5))
        else:
            elements.append(
                Paragraph("Local explanations could not be generated (version/shape mismatch).", self.style_n)
            )
        
        return elements
    
    def _get_top_features(self, global_exp) -> List[str]:
        """Extract top feature names from global explanation."""
        gd = global_exp.data()
        gnames = list(gd.get("names", []))
        gscores = np.asarray(gd.get("scores", []), dtype=float) if gd.get("scores", None) is not None else np.array([])
        
        top_feats: List[str] = []
        if len(gnames) == len(gscores) and len(gnames) > 0:
            order = np.argsort(np.abs(gscores))[::-1]
            # Filter out interaction terms (often appear as tuples/strings containing " & ")
            for idx in order:
                nm = gnames[idx]
                if isinstance(nm, str) and (" & " in nm or nm.startswith("(")):
                    continue
                if isinstance(nm, (tuple, list)):
                    continue
                top_feats.append(str(nm))
                if len(top_feats) >= self.config.top_n_features:
                    break
        
        return top_feats
    
    def _add_table(self, df: pd.DataFrame, max_rows: int = 20) -> List[Any]:
        """Add formatted table to report."""
        data = df_to_table_data(df, max_rows=max_rows)
        
        # Stringify cells for ReportLab
        data2 = []
        for row in data:
            formatted_row = []
            for cell in row:
                if isinstance(cell, float):
                    formatted_row.append(f"{cell:.6g}")
                else:
                    formatted_row.append(wrap_long(cell, self.config.wrap_length))
            data2.append(formatted_row)

        table = Table(data2, hAlign="LEFT")
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        return [table, Spacer(1, 6)]
    
    def _add_images_grid(
        self,
        image_paths: List[str],
        images_per_row: int = 2,
        img_width_cm: float = 8.0
    ) -> List[Any]:
        """Add grid of images to report."""
        elems: List[Any] = []
        row: List[Any] = []
        width = img_width_cm * cm
        max_height = 10 * cm  # Maximum height to prevent overflow

        for i, path in enumerate(image_paths, start=1):
            if not os.path.exists(path):
                self.logger.warning(f"Image not found: {path}")
                continue
            # Set both width and max height, preserving aspect ratio
            img = RLImage(path, width=width, height=max_height, kind='proportional')
            row.append(img)
            if len(row) == images_per_row:
                elems.append(Table([row], hAlign="LEFT"))
                elems.append(Spacer(1, 8))
                row = []

        if row:
            elems.append(Table([row], hAlign="LEFT"))
            elems.append(Spacer(1, 8))
        
        return elems
