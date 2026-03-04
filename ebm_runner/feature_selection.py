"""
Model-based feature subset selection for EBM models.

This module provides backward-elimination feature selection that iteratively
removes the least important feature and retrains the model, stopping when
performance degrades beyond a user-specified tolerance.
"""

import os
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import safe_json


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results.

    Attributes:
        selected_features: Final feature set after elimination.
        baseline_score: CV score using all features.
        final_score: CV score using the remaining feature subset.
        tolerance: The tolerance that was configured for elimination.
        history: DataFrame with one row per elimination step showing
            which feature was removed, the resulting CV score, and the
            delta from baseline.
    """

    selected_features: List[str]
    baseline_score: float
    final_score: float
    tolerance: float
    history: pd.DataFrame


class FeatureSelector:
    """Feature-elimination selector for EBM models.

    Given an ``EBMRunner`` instance (which controls hyper-parameter search,
    CV, and scoring settings), this class repeatedly:

    1. Fits an EBM on the current feature set.
    2. Ranks features by the model's ``term_importances()``.
    3. Removes the least important (backward) or most important (forward) feature.
    4. Stops when performance drops beyond *tolerance* or the feature
       count reaches *min_features*.
    """

    def __init__(
        self,
        runner: Any,  # EBMRunner – import kept lazy to avoid circular refs
        *,
        tolerance: float = 0.02,
        min_features: int = 1,
        step_percent: float = 0.0,
    ):
        """
        Args:
            runner: A configured ``EBMRunner`` instance.
            tolerance: Maximum acceptable relative drop in CV score
                compared to the baseline (all-features) score.
                E.g. 0.02 means a 2 % relative drop is tolerated.
            min_features: Never reduce below this many features.
            step_percent: Percentage of current features to remove at each step.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if min_features < 1:
            raise ValueError("min_features must be >= 1")
        if step_percent < 0.0 or step_percent >= 1.0:
            raise ValueError("step_percent must be in [0.0, 1.0)")

        self.runner = runner
        self.tolerance = tolerance
        self.min_features = min_features
        self.step_percent = step_percent
        self.logger: logging.Logger = runner.logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureSelectionResult:
        """Run backward-elimination feature selection.

        Args:
            X: Feature matrix (DataFrame or array).
            y: Target variable.
            feature_names: Required when *X* is an ndarray.

        Returns:
            A ``FeatureSelectionResult`` with the remaining feature subset.
        """
        from .utils import is_classification_target  # local import to keep top-level light

        label = "backward-elimination"

        start = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Starting {label} feature selection")
        self.logger.info(f"  tolerance = {self.tolerance}  |  min_features = {self.min_features}")
        self.logger.info("=" * 60)

        # --- prepare data ------------------------------------------------
        X_df, y_arr, feature_names = self.runner._prepare_data(X, y, feature_names)
        is_clf = is_classification_target(y_arr, self.runner.force_task)

        # --- Initial Tuning (if tune_once is enabled) --------------------
        self.fixed_params = None
        if self.runner.tune_once:
            self.logger.info("Tune-once enabled: Running hyperparameter search on full dataset...")
            search = self.runner._create_search(
                is_clf, y_arr, param_space=None, active_features=list(X_df.columns),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_df, y_arr)
            self.fixed_params = search.best_params_
            self.logger.info(f"Fixed parameters for feature selection: {safe_json(self.fixed_params)}")

        current_features: List[str] = list(feature_names)

        # --- baseline (all features) ------------------------------------
        baseline_score, baseline_model = self._fit_and_score(
            X_df[current_features], y_arr, is_clf
        )
        self.logger.info(f"Baseline CV score ({len(current_features)} features): {baseline_score:.6f}")

        history_rows: List[Dict[str, Any]] = [
            {
                "step": 0,
                "n_features": len(current_features),
                "removed_feature": None,
                "cv_score": baseline_score,
                "delta_from_baseline": 0.0,
                "relative_drop": 0.0,
            }
        ]

        best_features = list(current_features)
        best_score = baseline_score
        step = 0

        # --- iterative elimination --------------------------------------
        while len(current_features) > self.min_features:
            step += 1

            # Rank features by importance from the *current* model
            importances = self._get_importances(baseline_model if step == 1 else model, current_features)
            
            num_to_remove = 1
            if self.step_percent > 0:
                num_to_remove = max(1, int(len(current_features) * self.step_percent))
            
            # Ensure we don't go below min_features in one big jump
            num_to_remove = min(num_to_remove, len(current_features) - self.min_features)
            
            # backward -> remove least important features (bottom of the descending sorted list)
            target_features = importances["feature"].iloc[-num_to_remove:].tolist()
            
            candidate_features = [f for f in current_features if f not in target_features]
            target_features_str = ", ".join(target_features)
            
            self.logger.info(
                f"Step {step}: removing {len(target_features)} features: {target_features_str} "
                f"({len(candidate_features)} features remaining)"
            )

            score, model = self._fit_and_score(X_df[candidate_features], y_arr, is_clf)

            # For maximised metrics the score is positive; we want to
            # detect a *drop*.  For neg_* metrics (e.g. neg_rmse) the
            # score is negative, so a "drop" means the value became
            # more negative.  In both cases, checking
            #   (baseline - score) / |baseline|  >  tolerance
            # captures a worsening.
            abs_baseline = abs(baseline_score) if abs(baseline_score) > 1e-12 else 1e-12
            relative_drop = (baseline_score - score) / abs_baseline

            delta = score - baseline_score

            history_rows.append(
                {
                    "step": step,
                    "n_features": len(candidate_features),
                    "removed_feature": target_features_str,
                    "cv_score": score,
                    "delta_from_baseline": delta,
                    "relative_drop": relative_drop,
                }
            )

            self.logger.info(
                f"  CV score: {score:.6f}  |  "
                f"delta: {delta:+.6f}  |  "
                f"relative drop: {relative_drop:.4f}"
            )

            if relative_drop > self.tolerance:
                self.logger.info(
                    f"  Stopping: relative drop {relative_drop:.4f} > "
                    f"tolerance {self.tolerance:.4f}"
                )
                break

            # Accept the removal
            current_features = candidate_features
            best_features = list(current_features)
            best_score = score

        elapsed = time.time() - start
        self.logger.info("-" * 60)
        self.logger.info(
            f"Feature selection complete in {elapsed:.1f}s  |  "
            f"{len(best_features)}/{len(feature_names)} features retained"
        )
        self.logger.info(f"Selected features: {best_features}")
        self.logger.info(
            f"Baseline score: {baseline_score:.6f}  |  "
            f"Final score: {best_score:.6f}"
        )

        history_df = pd.DataFrame(history_rows)

        out_path = os.path.join(self.runner.output_dir, "feature_selection_summary.csv")
        try:
            history_df.to_csv(out_path, index=False)
            self.logger.info(f"Saved feature selection summary to {out_path}")
        except Exception as e:
            self.logger.warning(f"Could not save feature selection summary: {e}")

        return FeatureSelectionResult(
            selected_features=best_features,
            baseline_score=baseline_score,
            final_score=best_score,
            tolerance=self.tolerance,
            history=history_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_and_score(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        is_clf: bool,
    ) -> tuple:
        """Fit an EBM via the runner's CV/search pipeline and return (best_cv_score, best_model)."""
        active_features = list(X.columns)

        # Optimization: Reuse fixed params if tune_once is active
        if getattr(self, "fixed_params", None) is not None:
            estimator = self.runner._make_estimator(is_clf, active_features=active_features)
            estimator.set_params(**self.fixed_params)

            # Evaluate via cross-validation with the fixed hyperparameters
            cv = self.runner._make_cv(is_clf, y)
            scoring = self.runner._make_scoring(is_clf, y)

            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                estimator, X, y,
                cv=cv, scoring=scoring, n_jobs=self.runner.n_jobs
            )
            mean_score = scores.mean()

            # Fit on the full subset to get importances for the next step
            estimator.fit(X, y)

            return mean_score, estimator

        # Normal full search
        search = self.runner._create_search(
            is_clf, y, param_space=None, active_features=active_features,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        return search.best_score_, search.best_estimator_

    def _get_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Return a DataFrame of (feature, importance) sorted descending by importance.

        Uses ``term_importances()`` which returns one importance value per
        term (feature or interaction).  We keep only the single-feature
        terms that are in *feature_names* and ignore interaction terms.
        """
        try:
            importances = model.term_importances()
            term_names = model.term_names_

            rows = []
            for name, imp in zip(term_names, importances):
                # Skip interaction terms (they contain " & " or are tuples)
                if isinstance(name, (tuple, list)):
                    continue
                if isinstance(name, str) and " & " in name:
                    continue
                if name in feature_names:
                    rows.append({"feature": name, "importance": float(imp)})

            df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.warning(f"term_importances() failed ({e}), falling back to explain_global()")
            return self._get_importances_fallback(model, feature_names)

    def _get_importances_fallback(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Fallback: extract importances from explain_global()."""
        global_exp = model.explain_global()
        gd = global_exp.data()
        names = list(gd.get("names", []))
        scores = np.asarray(gd.get("scores", []), dtype=float)

        rows = []
        for name, score in zip(names, scores):
            if isinstance(name, (tuple, list)):
                continue
            if isinstance(name, str) and " & " in name:
                continue
            if name in feature_names:
                rows.append({"feature": name, "importance": float(abs(score))})

        df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
