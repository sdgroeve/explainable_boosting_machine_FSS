"""
Main EBM Runner class for model training, tuning, and evaluation.

This module provides the EBMRunner class which orchestrates the complete
workflow of training, hyperparameter tuning, evaluation, and reporting
for Explainable Boosting Machine models.
"""

import os
import time
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_predict,
    cross_val_score,
)

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
except ImportError as e:
    raise ImportError(
        "Could not import interpret.glassbox EBM models. "
        "Install with: pip install interpret"
    ) from e

from .config import RunConfig, PlotConfig, ReportConfig
from .utils import (
    make_logger,
    ensure_dir,
    safe_json,
    is_classification_target,
    validate_input_data,
)
from .metrics import ValidationResult, compute_classification_metrics, compute_regression_metrics
from .html_reporting import HTMLReportGenerator


@dataclass
class EBMRunArtifacts:
    """Container for all artifacts produced by an EBM run."""
    
    best_model: Any
    is_classification: bool
    best_params: Dict[str, Any]
    cv_results: Optional[pd.DataFrame]
    test_result: ValidationResult
    report_html_path: str
    model_path: str
    cv_results_csv_path: Optional[str]
    eval_strategy: str = "train_test"


class EBMRunner:
    """
    Main runner for EBM model training, tuning, and evaluation.
    
    This class handles the complete workflow:
    1. Data validation and splitting
    2. Hyperparameter tuning via cross-validation
    3. Model evaluation on holdout test set
    4. PDF report generation with interpretability plots
    5. Artifact persistence (model, results, plots)
    """
    
    def __init__(
        self,
        *,
        output_dir: str = RunConfig.output_dir,
        random_state: int = RunConfig.random_state,
        force_task: Optional[str] = None,
        test_size: float = RunConfig.test_size,
        val_size: float = RunConfig.val_size,
        n_splits: Optional[int] = None,
        n_iter: int = RunConfig.n_iter,
        use_grid: bool = RunConfig.use_grid,
        n_jobs: int = RunConfig.n_jobs,
        ebm_n_jobs: int = RunConfig.ebm_n_jobs,
        top_n_features_in_report: int = 12,
        n_local_samples_in_report: int = 3,
        enable_interactions: bool = RunConfig.enable_interactions,
        eval_strategy: str = RunConfig.eval_strategy,
        tuning_n_splits: int = RunConfig.tuning_n_splits,
        tuning_stratified: bool = RunConfig.tuning_stratified,
        eval_n_splits: int = RunConfig.eval_n_splits,
        eval_stratified: bool = RunConfig.eval_stratified,
        tune_once: bool = RunConfig.tune_once,
        tuning_verbose: int = RunConfig.tuning_verbose,
        tuning_show_all: bool = RunConfig.tuning_show_all,
        explain_positive_class: bool = RunConfig.explain_positive_class,
        save_predictions: bool = RunConfig.save_predictions,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize EBM Runner.
        
        Args:
            output_dir: Directory for outputs (model, plots, reports)
            random_state: Random seed for reproducibility
            force_task: Force task type ('classification' or 'regression'), or None for auto-detect
            test_size: Fraction of data for holdout test set
            val_size: Fraction of training data for validation (0 = use CV only)
            n_splits: Shorthand that sets both tuning_n_splits and eval_n_splits
            n_iter: Number of iterations for RandomizedSearchCV
            use_grid: Use GridSearchCV instead of RandomizedSearchCV
            n_jobs: Number of parallel jobs for search (-1 = all cores)
            ebm_n_jobs: Number of parallel jobs for EBM training
            top_n_features_in_report: Number of top features to show in report
            n_local_samples_in_report: Number of local explanations in report
            enable_interactions: Enable interaction terms in hyperparameter search
            eval_strategy: 'train_test', 'cv_only', or 'train_only'
            tuning_n_splits: CV folds for hyperparameter search
            tuning_stratified: Stratify tuning folds for classification
            eval_n_splits: CV folds for cv_only evaluation
            eval_stratified: Stratify eval folds for classification
            tune_once: If True, tune hyperparameters once on full features
            feature_names: Optional list of feature names for EBM
            feature_types: Optional list of feature types for EBM
            logger: Optional logger instance
        """
        if eval_strategy not in ("train_test", "cv_only", "train_only"):
            raise ValueError(
                f"eval_strategy must be 'train_test', 'cv_only', or 'train_only', "
                f"got {eval_strategy!r}"
            )

        self.output_dir = output_dir
        self.random_state = random_state
        self.force_task = force_task
        self.test_size = test_size
        self.val_size = val_size
        # n_splits is a convenience shorthand that sets both tuning and eval
        if n_splits is not None:
            tuning_n_splits = n_splits
            eval_n_splits = n_splits
        self.tuning_n_splits = tuning_n_splits
        self.tuning_stratified = tuning_stratified
        self.eval_n_splits = eval_n_splits
        self.eval_stratified = eval_stratified
        self.n_iter = n_iter
        self.use_grid = use_grid
        self.n_jobs = n_jobs
        self.ebm_n_jobs = ebm_n_jobs
        self.enable_interactions = enable_interactions
        self.eval_strategy = eval_strategy
        self.tune_once = tune_once
        self.tuning_verbose = tuning_verbose
        self.tuning_show_all = tuning_show_all
        self.explain_positive_class = explain_positive_class
        self.save_predictions = save_predictions
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.logger = logger or make_logger()

        # Setup directories
        ensure_dir(self.output_dir)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        ensure_dir(self.plots_dir)
        
        # Setup report configuration
        report_config = ReportConfig(
            top_n_features=top_n_features_in_report,
            n_local_samples=n_local_samples_in_report,
        )
        self.report_generator = HTMLReportGenerator(
            config=report_config,
            logger=self.logger,
        )

    def fit_optimize_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        feature_names: Optional[List[str]] = None,
        param_space: Optional[Dict[str, Any]] = None,
        report_name: str = "ebm_report.html",
        model_name: str = "best_ebm_model.joblib",
        feature_selection_result: Optional[Any] = None,
    ) -> EBMRunArtifacts:
        """
        Complete EBM workflow: train, tune, evaluate, and report.
        
        The evaluation strategy is controlled by ``self.eval_strategy``:

        - ``train_test`` – split data, CV-tune on train, evaluate on holdout.
        - ``cv_only``    – CV-tune on all data; report shows CV scores.
        - ``train_only`` – single fit on all data; evaluate on training set.
        """
        self.logger.info(f"Starting EBM run (eval_strategy={self.eval_strategy})")
        start_time = time.time()

        # Prepare and validate data
        X_df, y_arr, feature_names = self._prepare_data(X, y, feature_names)

        # Detect task type
        is_clf = is_classification_target(y_arr, self.force_task)
        self.logger.info(f"Detected task: {'classification' if is_clf else 'regression'}")

        # ── Dispatch to the chosen strategy ──────────────────────────
        if self.eval_strategy == "train_test":
            artifacts = self._run_train_test(
                X_df, y_arr, is_clf, feature_names, param_space,
                report_name, model_name,
                feature_selection_result,
            )
        elif self.eval_strategy == "cv_only":
            artifacts = self._run_cv_only(
                X_df, y_arr, is_clf, feature_names, param_space,
                report_name, model_name,
                feature_selection_result,
            )
        else:  # train_only
            artifacts = self._run_train_only(
                X_df, y_arr, is_clf, feature_names, param_space,
                report_name, model_name,
                feature_selection_result,
            )

        elapsed = time.time() - start_time
        self.logger.info(f"Done. Total elapsed: {elapsed:.1f}s")
        return artifacts

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _run_train_test(
        self, X_df, y_arr, is_clf, feature_names, param_space,
        report_name, model_name,
        feature_selection_result,
    ) -> EBMRunArtifacts:
        """Strategy: train/test split → CV-tune on train → evaluate on holdout."""
        # Preserve original index so we can write predictions in dataset order
        X_indexed = X_df.reset_index(drop=False)  # keeps original index as a column if needed
        X_train, X_test, y_train, y_test = self._split_data(X_df, y_arr, is_clf)
        test_indices = X_test.index  # original positions in X_df

        search = self._create_search(is_clf, y_train, param_space,
                                      active_features=list(X_train.columns))
        best_model = self._run_hyperparameter_search(search, X_train, y_train)

        y_pred, y_proba, test_result = self._evaluate_model(
            best_model, X_test, y_test, is_clf
        )

        # Save predictions CSV (in original dataset order)
        if self.save_predictions:
            self._save_predictions_csv(
                original_indices=test_indices,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                is_clf=is_clf,
            )

        # Positive-class rows for local explanations  (predicted label == 1)
        explain_positive_X = None
        explain_positive_y = None
        if self.explain_positive_class and is_clf and feature_selection_result is None:
            pos_mask = y_pred == 1
            if pos_mask.any():
                explain_positive_X = X_test[pos_mask]
                explain_positive_y = y_test[pos_mask]

        model_path, cv_csv = self._save_artifacts(best_model, search, model_name)

        html_path = os.path.join(self.output_dir, report_name)
        self.report_generator.generate_report(
            html_path=html_path,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            best_model=best_model,
            is_classification=is_clf,
            best_params=search.best_params_,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            y_pred=y_pred, y_proba=y_proba,
            feature_selection_result=feature_selection_result,
            eval_strategy=self.eval_strategy,
            explain_positive_X=explain_positive_X,
            explain_positive_y=explain_positive_y,
        )

        return EBMRunArtifacts(
            best_model=best_model,
            is_classification=is_clf,
            best_params=search.best_params_,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            test_result=test_result,
            report_html_path=html_path,
            model_path=model_path,
            cv_results_csv_path=cv_csv,
            eval_strategy=self.eval_strategy,
        )

    def _run_cv_only(
        self, X_df, y_arr, is_clf, feature_names, param_space,
        report_name, model_name,
        feature_selection_result,
    ) -> EBMRunArtifacts:
        """Strategy: CV-tune on all data, no holdout."""
        self.logger.info(
            f"CV-only mode: using all {len(X_df)} samples  |  "
            f"tuning: {self.tuning_n_splits}-fold  |  eval: {self.eval_n_splits}-fold"
        )

        # Run hyper-parameter search on the full dataset
        search = self._create_search(is_clf, y_arr, param_space,
                                      active_features=list(X_df.columns))
        best_model = self._run_hyperparameter_search(search, X_df, y_arr)

        # Collect per-fold CV scores as the evaluation metric
        eval_cv = self._make_cv(
            is_clf, y_arr,
            n_splits=self.eval_n_splits,
            stratified=self.eval_stratified,
        )
        scoring = self._make_scoring(is_clf, y_arr)
        fold_scores = cross_val_score(
            best_model, X_df, y_arr, cv=eval_cv, scoring=scoring, n_jobs=self.n_jobs,
        )
        self.logger.info(
            f"CV fold scores: {fold_scores}  |  mean={fold_scores.mean():.6f}  "
            f"std={fold_scores.std():.6f}"
        )

        # Build metrics dict from CV results
        metrics: Dict[str, Any] = {
            "cv_mean_score": float(fold_scores.mean()),
            "cv_std_score": float(fold_scores.std()),
        }
        for i, s in enumerate(fold_scores):
            metrics[f"fold_{i+1}_score"] = float(s)

        test_result = ValidationResult(metrics=metrics, extra={"fold_scores": fold_scores.tolist()})

        # Generate out-of-fold predictions for the report plots
        y_pred = cross_val_predict(best_model, X_df, y_arr, cv=eval_cv, n_jobs=self.n_jobs)
        y_proba = None
        if is_clf and hasattr(best_model, "predict_proba"):
            try:
                y_proba = cross_val_predict(
                    best_model, X_df, y_arr, cv=eval_cv, method="predict_proba",
                    n_jobs=self.n_jobs,
                )
            except Exception as e:
                self.logger.warning(f"Could not get CV predicted probabilities: {e}")

        # Save predictions CSV (OOF predictions, already in original order)
        if self.save_predictions:
            self._save_predictions_csv(
                original_indices=X_df.index,
                y_true=y_arr,
                y_pred=y_pred,
                y_proba=y_proba,
                is_clf=is_clf,
            )

        # Positive-class rows for local explanations (predicted label == 1)
        explain_positive_X = None
        explain_positive_y = None
        if self.explain_positive_class and is_clf and feature_selection_result is None:
            pos_mask = y_pred == 1
            if pos_mask.any():
                explain_positive_X = X_df[pos_mask]
                explain_positive_y = y_arr[pos_mask]

        # Refit the final model on all data
        best_model.fit(X_df, y_arr)

        model_path, cv_csv = self._save_artifacts(best_model, search, model_name)

        html_path = os.path.join(self.output_dir, report_name)
        self.report_generator.generate_report(
            html_path=html_path,
            X_train=X_df, y_train=y_arr,
            X_test=X_df, y_test=y_arr,
            best_model=best_model,
            is_classification=is_clf,
            best_params=search.best_params_,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            y_pred=y_pred, y_proba=y_proba,
            feature_selection_result=feature_selection_result,
            eval_strategy=self.eval_strategy,
            explain_positive_X=explain_positive_X,
            explain_positive_y=explain_positive_y,
        )

        return EBMRunArtifacts(
            best_model=best_model,
            is_classification=is_clf,
            best_params=search.best_params_,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            test_result=test_result,
            report_html_path=html_path,
            model_path=model_path,
            cv_results_csv_path=cv_csv,
            eval_strategy=self.eval_strategy,
        )

    def _run_train_only(
        self, X_df, y_arr, is_clf, feature_names, param_space,
        report_name, model_name,
        feature_selection_result,
    ) -> EBMRunArtifacts:
        """Strategy: tune via CV on all data, refit on all data, evaluate on training set."""
        self.logger.info(
            f"Train-only mode: {len(X_df)} samples  |  "
            f"tuning with {self.tuning_n_splits}-fold CV"
        )
        self.logger.warning(
            "Metrics are computed on the training data and will be optimistic. "
            "Use with caution."
        )

        search = self._create_search(is_clf, y_arr, param_space,
                                      active_features=list(X_df.columns))
        best_model = self._run_hyperparameter_search(search, X_df, y_arr)

        best_params = search.best_params_
        self.logger.info(f"Best params: {best_params}")

        # Refit best model on all data
        best_model.fit(X_df, y_arr)

        y_pred, y_proba, test_result = self._evaluate_model(
            best_model, X_df, y_arr, is_clf
        )

        # Save predictions CSV (train predictions in original order)
        if self.save_predictions:
            self._save_predictions_csv(
                original_indices=X_df.index,
                y_true=y_arr,
                y_pred=y_pred,
                y_proba=y_proba,
                is_clf=is_clf,
            )

        # Positive-class rows for local explanations
        explain_positive_X = None
        explain_positive_y = None
        if self.explain_positive_class and is_clf and feature_selection_result is None:
            pos_mask = y_pred == 1
            if pos_mask.any():
                explain_positive_X = X_df[pos_mask]
                explain_positive_y = y_arr[pos_mask]

        model_path, cv_csv = self._save_artifacts(best_model, search, model_name)

        html_path = os.path.join(self.output_dir, report_name)
        self.report_generator.generate_report(
            html_path=html_path,
            X_train=X_df, y_train=y_arr,
            X_test=X_df, y_test=y_arr,
            best_model=best_model,
            is_classification=is_clf,
            best_params=best_params,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            y_pred=y_pred, y_proba=y_proba,
            feature_selection_result=feature_selection_result,
            eval_strategy=self.eval_strategy,
            explain_positive_X=explain_positive_X,
            explain_positive_y=explain_positive_y,
        )

        return EBMRunArtifacts(
            best_model=best_model,
            is_classification=is_clf,
            best_params=best_params,
            cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
            test_result=test_result,
            report_html_path=html_path,
            model_path=model_path,
            cv_results_csv_path=cv_csv,
            eval_strategy=self.eval_strategy,
        )

    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]]
    ) -> tuple:
        """Prepare and validate input data."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            if feature_names is None:
                feature_names = list(X_df.columns)

        y_arr = np.asarray(y)
        
        # Validate inputs
        validate_input_data(X_df, y_arr, feature_names, self.logger)
        
        return X_df, y_arr, feature_names

    def _split_data(
        self,
        X_df: pd.DataFrame,
        y_arr: np.ndarray,
        is_clf: bool
    ) -> tuple:
        """Split data into train and test sets."""
        stratify = y_arr if is_clf else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        self.logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

        # Optional separate validation split (rarely needed with CV)
        if self.val_size and self.val_size > 0:
            stratify2 = y_train if is_clf else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=stratify2
            )
            self.logger.info(f"Extra validation split enabled. Val size: {len(X_val)}")

        return X_train, X_test, y_train, y_test

    def _create_search(
        self,
        is_clf: bool,
        y_train: np.ndarray,
        param_space: Optional[Dict[str, Any]],
        active_features: Optional[List[str]] = None,
    ):
        """Create hyperparameter search object."""
        estimator = self._make_estimator(is_clf, active_features=active_features)
        space = param_space or self._default_param_space(is_clf)
        cv = self._make_cv(
            is_clf, y_train,
            n_splits=self.tuning_n_splits,
            stratified=self.tuning_stratified,
        )
        scoring = self._make_scoring(is_clf, y_train)

        # ── Search summary header ─────────────────────────────────────
        method = "GridSearchCV" if self.use_grid else "RandomizedSearchCV"
        n_candidates = (
            int(np.prod([len(v) if hasattr(v, '__len__') else 1 for v in space.values()]))
            if self.use_grid
            else self.n_iter
        )
        n_folds = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else self.tuning_n_splits
        total_fits = n_candidates * n_folds

        self.logger.info(
            f"Hyperparameter search: {method} | "
            f"{n_candidates} candidates × {n_folds}-fold CV = {total_fits} fits | "
            f"scoring={scoring}"
        )
        space_desc = ", ".join(
            f"{k} ({len(v)} values)" if hasattr(v, '__len__') else f"{k} (1 value)"
            for k, v in space.items()
        )
        self.logger.info(f"Search space: {space_desc}")

        if self.use_grid:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=space,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.tuning_verbose,
                refit=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=space,
                n_iter=self.n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.tuning_verbose,
                refit=True,
            )
        
        return search

    def _run_hyperparameter_search(self, search, X_train, y_train):
        """Run hyperparameter search and return best model."""
        self.logger.info("Fitting hyperparameter search …")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        elapsed = time.time() - t0

        # ── Elapsed time ──────────────────────────────────────────────
        mins, secs = divmod(int(elapsed), 60)
        self.logger.info(
            f"Hyperparameter search completed in "
            f"{f'{mins}m ' if mins else ''}{secs}s"
        )

        # ── Candidates table ──────────────────────────────────────────
        n_show = None if self.tuning_show_all else 10
        cv_df = (
            pd.DataFrame(search.cv_results_)
            .sort_values("rank_test_score")
        )
        if n_show:
            cv_df = cv_df.head(n_show)
            label = f"Top {len(cv_df)} candidates"
        else:
            label = f"All {len(cv_df)} candidates"
        param_cols = [c for c in cv_df.columns if c.startswith("param_")]
        self.logger.info(f"── {label} " + "─" * (65 - len(label)))
        for _, row in cv_df.iterrows():
            rank = int(row["rank_test_score"])
            mean = row["mean_test_score"]
            std  = row["std_test_score"]
            params = "  ".join(
                f"{c.replace('param_', '')}={row[c]}"
                for c in param_cols
            )
            self.logger.info(
                f"  #{rank:<3d} mean={mean:.6f}  std={std:.6f}  {params}"
            )

        self.logger.info(f"Best CV score: {search.best_score_:.6f}")
        self.logger.info(f"Best params: {safe_json(search.best_params_)}")

        return search.best_estimator_

    def _evaluate_model(self, best_model, X_test, y_test, is_clf):
        """Evaluate model on test set."""
        self.logger.info("Evaluating on holdout test set...")
        y_pred = best_model.predict(X_test)

        y_proba = None
        if is_clf and hasattr(best_model, "predict_proba"):
            try:
                y_proba = best_model.predict_proba(X_test)
            except Exception as e:
                self.logger.warning(f"Could not get predicted probabilities: {e}")
                y_proba = None

        if is_clf:
            metrics = compute_classification_metrics(y_test, y_pred, y_proba, self.logger)
        else:
            metrics = compute_regression_metrics(y_test, y_pred, self.logger)

        test_result = ValidationResult(metrics=metrics, extra={})
        self.logger.info(f"Holdout metrics: {safe_json(metrics)}")

        return y_pred, y_proba, test_result

    def _save_artifacts(self, best_model, search, model_name):
        """Save model and CV results."""
        model_path = os.path.join(self.output_dir, model_name)
        dump(best_model, model_path)
        self.logger.info(f"Saved best model: {model_path}")

        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv_csv = os.path.join(self.output_dir, "cv_results.csv")
        cv_results.to_csv(cv_csv, index=False)
        self.logger.info(f"Saved CV results: {cv_csv}")

        return model_path, cv_csv

    def _save_predictions_csv(
        self,
        original_indices,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        is_clf: bool,
    ) -> None:
        """Write per-datapoint predictions to predictions.csv in original dataset order.

        Columns: original_index, y_true, y_pred, and (binary classification only)
        prob_positive.
        """
        df = pd.DataFrame({
            "original_index": original_indices,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if is_clf and y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            df["prob_positive"] = y_proba[:, 1]
        path = os.path.join(self.output_dir, "predictions.csv")
        df.to_csv(path, index=False)
        self.logger.info(f"Saved predictions CSV: {path}")

    def _default_param_space(self, is_classification: bool) -> Dict[str, Any]:
        """
        Get default hyperparameter search space for EBM.
        
        Args:
            is_classification: Whether task is classification
            
        Returns:
            Dictionary of hyperparameter distributions
        """
        space = {
            "learning_rate": np.linspace(0.01, 0.2, 20),
            "max_bins": [64, 128, 256],
            "max_leaves": [2, 3, 4, 5, 6, 8],
            "min_samples_leaf": [1, 2, 5, 10, 20, 50],
            "max_rounds": [200, 400, 800, 1200],
            "interactions": [0, 5, 10, 20] if self.enable_interactions else [0],
            "outer_bags": [4, 8],
        }
        return space

    def _make_estimator(self, is_classification: bool, *, active_features: Optional[List[str]] = None):
        """Create EBM estimator.

        Args:
            is_classification: Whether this is a classification task.
            active_features: When provided (e.g. during feature selection),
                only these features (a subset of ``self.feature_names``)
                are passed to the EBM so that the name/type lists match
                the columns of the data.
        """
        kwargs = dict(
            random_state=self.random_state,
            n_jobs=self.ebm_n_jobs,
        )

        # Determine the feature_names / feature_types to pass.
        # If active_features is given (feature selection is narrowing the
        # data), we filter to keep only those features.
        feat_names = self.feature_names
        feat_types = self.feature_types

        if active_features is not None and feat_names is not None:
            # Build an index from original name → position so we can
            # look up the matching type for each active feature.
            name_to_idx = {n: i for i, n in enumerate(feat_names)}
            indices = [name_to_idx[f] for f in active_features if f in name_to_idx]
            feat_names = [feat_names[i] for i in indices]
            if feat_types is not None:
                feat_types = [feat_types[i] for i in indices]

        if feat_names is not None:
            kwargs["feature_names"] = feat_names
        if feat_types is not None:
            kwargs["feature_types"] = feat_types

        if is_classification:
            return ExplainableBoostingClassifier(**kwargs)
        return ExplainableBoostingRegressor(**kwargs)

    def _make_cv(
        self,
        is_classification: bool,
        y_train: np.ndarray,
        *,
        n_splits: Optional[int] = None,
        stratified: Optional[bool] = None,
    ):
        """Create cross-validation splitter.

        Args:
            n_splits: Number of folds (defaults to tuning_n_splits).
            stratified: Whether to stratify (defaults to tuning_stratified).
        """
        if n_splits is None:
            n_splits = self.tuning_n_splits
        if stratified is None:
            stratified = self.tuning_stratified

        if is_classification and stratified:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    def _make_scoring(self, is_classification: bool, y: np.ndarray) -> str:
        """Determine scoring metric for CV."""
        if not is_classification:
            return "neg_root_mean_squared_error"
        # Binary vs multiclass
        classes = np.unique(y)
        if len(classes) <= 2:
            return "roc_auc"
        return "roc_auc_ovr"

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        tolerance: float = 0.02,
        min_features: int = 1,
        step_percent: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureSelectionResult":
        """Find a minimal feature subset with near-equivalent performance.

        Uses backward elimination: starting from all features, the least
        important feature (according to the EBM's ``term_importances()``)
        is removed at each step.  The model is retrained and the CV score
        is compared to the baseline.  Elimination stops when the relative
        performance drop exceeds *tolerance* or *min_features* is reached.

        Args:
            X: Feature matrix (DataFrame or array).
            y: Target variable (Series or array).
            tolerance: Maximum acceptable *relative* drop in CV score
                compared to the all-features baseline (e.g. 0.02 = 2%).
            min_features: Never reduce below this many features.
            step_percent: Percentage of current features to remove at each step.
            feature_names: Optional feature names (required if X is array).

        Returns:
            A ``FeatureSelectionResult`` with the selected features, scores,
            and a step-by-step elimination history.
        """
        from .feature_selection import FeatureSelector

        selector = FeatureSelector(
            runner=self,
            tolerance=tolerance,
            min_features=min_features,
            step_percent=step_percent,
        )
        return selector.select_features(X, y, feature_names=feature_names)


