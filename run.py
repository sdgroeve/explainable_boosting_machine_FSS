"""
EBM Runner – command-line entry point.

Runs a classification or regression demo with optional feature selection.
Feature selection results (if enabled) are included in the HTML report.

Usage examples
--------------
  # Default demo (classification + regression, no feature selection)
  python run.py

  # Classification only, with feature selection (2 % tolerance, min 3 features)
  python run.py --task clf --feature-selection --fs-tolerance 0.02 --fs-min-features 3

  # Regression only, with feature selection
  python run.py --task reg --feature-selection

  # Load a CSV dataset and run classification
  python run.py --data mydata.csv --target y_col --task clf

  # Load a CSV dataset with a feature-types file
  python run.py --data mydata.csv --target y_col --feature-types types.csv --task clf
"""

import argparse
import logging
import sys
from typing import List, Optional, Tuple, Union

import pandas as pd

from sklearn.datasets import load_breast_cancer, load_diabetes

from ebm_runner import EBMRunner
from ebm_runner.config import RunConfig
from ebm_runner.utils import make_logger

_DEFAULTS = RunConfig()


# ── helpers ──────────────────────────────────────────────────────────

def _parse_feature_type(raw: str) -> Union[str, List[str]]:
    """Parse a single feature-type value from the types file.

    Values that look like bracket-delimited lists (e.g. ``"[1,2,3+]"``)
    are converted to a real Python list of strings.  All other values are
    returned as-is (e.g. ``"continuous"``, ``"nominal"``).
    """
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        # Split on commas, strip whitespace from each element
        return [item.strip() for item in inner.split(",") if item.strip()]
    return stripped


def _load_feature_types(
    path: str,
) -> Tuple[List[str], list]:
    """Load a feature-types CSV file.

    The file must have two columns: ``column`` and ``type``.  The ``type``
    column may contain plain strings (e.g. ``"continuous"``) or
    bracket-delimited lists formatted as strings (e.g. ``"[1,2,3+]"``),
    which are converted to real Python lists of strings.

    Returns:
        A tuple of ``(feature_names, feature_types)`` where
        *feature_names* is the list of column names and *feature_types*
        is the list of parsed type values.
    """
    df = pd.read_csv(path)

    # Validate expected columns
    for col in ("feature", "type"):
        if col not in df.columns:
            raise ValueError(
                f"Feature-types file {path!r} is missing required column {col!r}. "
                f"Found columns: {list(df.columns)}"
            )

    feature_names = df["feature"].astype(str).tolist()
    feature_types = [_parse_feature_type(v) for v in df["type"].astype(str)]

    return feature_names, feature_types


def _run_from_csv(
    logger: logging.Logger,
    *,
    data_path: str,
    target_column: str,
    feature_types_path: Optional[str] = None,
    task: str = "clf",
    feature_selection: bool = False,
    forward_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
) -> None:
    """Run the EBM pipeline on a user-supplied CSV dataset."""
    logger.info("=" * 60)
    logger.info(f"Running from CSV: {data_path}")
    logger.info("=" * 60)

    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Drop rows where target is missing
    missing_mask = y.isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        logger.warning(
            f"Dropping {n_missing} row(s) with missing target values "
            f"({n_missing / len(y) * 100:.1f}% of data)"
        )
        X = X[~missing_mask].reset_index(drop=True)
        y = y[~missing_mask].reset_index(drop=True)

    # Load optional feature types
    ebm_feature_names = None
    ebm_feature_types = None
    if feature_types_path:
        ebm_feature_names, ebm_feature_types = _load_feature_types(feature_types_path)
        logger.info(
            f"Loaded feature types for {len(ebm_feature_names)} features "
            f"from {feature_types_path}"
        )

    force_task = "classification" if task == "clf" else "regression"
    runner = EBMRunner(
        output_dir=f"./ebm_output",
        n_iter=30,
        enable_interactions=True,
        force_task=force_task,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        feature_names=ebm_feature_names,
        feature_types=ebm_feature_types,
        logger=logger,
    )

    # Optional backward feature selection
    fs_result = None
    if feature_selection:
        fs_result = runner.select_features(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    # Optional forward (redundancy) feature selection
    fwd_result = None
    if forward_selection:
        fwd_result = runner.select_features_forward(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Forward: {len(fwd_result.selected_features)}/{X.shape[1]} features remaining: "
            f"{fwd_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_report.html",
        feature_selection_result=fs_result,
        forward_selection_result=fwd_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


def _run_classification(
    logger: logging.Logger,
    *,
    feature_selection: bool = False,
    forward_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
) -> None:
    """Run the breast-cancer classification demo."""
    logger.info("=" * 60)
    logger.info("Classification Example (Breast Cancer)")
    logger.info("=" * 60)

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    runner = EBMRunner(
        output_dir="./ebm_breast_cancer",
        n_iter=3,
        enable_interactions=True,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        logger=logger,
    )

    # Optional backward feature selection
    fs_result = None
    if feature_selection:
        fs_result = runner.select_features(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    # Optional forward (redundancy) feature selection
    fwd_result = None
    if forward_selection:
        fwd_result = runner.select_features_forward(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Forward: {len(fwd_result.selected_features)}/{X.shape[1]} features remaining: "
            f"{fwd_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_classification_report.html",
        feature_selection_result=fs_result,
        forward_selection_result=fwd_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


def _run_regression(
    logger: logging.Logger,
    *,
    feature_selection: bool = False,
    forward_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
) -> None:
    """Run the diabetes regression demo."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Regression Example (Diabetes)")
    logger.info("=" * 60)

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    runner = EBMRunner(
        output_dir="./ebm_diabetes",
        n_iter=30,
        enable_interactions=True,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        logger=logger,
    )

    # Optional backward feature selection
    fs_result = None
    if feature_selection:
        fs_result = runner.select_features(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    # Optional forward (redundancy) feature selection
    fwd_result = None
    if forward_selection:
        fwd_result = runner.select_features_forward(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
        )
        logger.info(
            f"Forward: {len(fwd_result.selected_features)}/{X.shape[1]} features remaining: "
            f"{fwd_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_regression_report.html",
        feature_selection_result=fs_result,
        forward_selection_result=fwd_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


# ── CLI ──────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EBM model training with optional feature selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg", "all"],
        default="all",
        help="Which task to run (default: all). 'all' runs both demo datasets when --data is not set.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a CSV dataset file. When provided, --target is required.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target column in the CSV dataset.",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        default=None,
        help=(
            "Path to a CSV file with columns 'column' and 'type'. "
            "The 'type' column may contain plain strings (e.g. 'continuous') "
            "or bracket-delimited lists (e.g. '[1,2,3+]')."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        default=False,
        help="Enable backward-elimination feature selection.",
    )
    parser.add_argument(
        "--fs-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative CV-score drop allowed during elimination (default: 0.02 = 2%%).",
    )
    parser.add_argument(
        "--fs-min-features",
        type=int,
        default=3,
        help="Minimum number of features to keep (default: 3).",
    )
    parser.add_argument(
        "--forward-selection",
        action="store_true",
        default=False,
        help="Enable forward-elimination (redundancy analysis) feature selection.",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=["train_test", "cv_only", "train_only"],
        default=_DEFAULTS.eval_strategy,
        help=(
            f"Evaluation strategy (default: {_DEFAULTS.eval_strategy}). "
            "'train_test' splits data into train/test; "
            "'cv_only' uses all data with cross-validation; "
            "'train_only' fits on all data without splitting or tuning."
        ),
    )
    parser.add_argument(
        "--tuning-folds",
        type=int,
        default=_DEFAULTS.tuning_n_splits,
        help=f"CV folds for hyperparameter tuning (default: {_DEFAULTS.tuning_n_splits}).",
    )
    parser.add_argument(
        "--eval-folds",
        type=int,
        default=_DEFAULTS.eval_n_splits,
        help=f"CV folds for cv_only evaluation (default: {_DEFAULTS.eval_n_splits}).",
    )
    parser.add_argument(
        "--no-tuning-stratify",
        action="store_true",
        default=not _DEFAULTS.tuning_stratified,
        help="Disable stratified CV for tuning (use plain KFold).",
    )
    parser.add_argument(
        "--no-eval-stratify",
        action="store_true",
        default=not _DEFAULTS.eval_stratified,
        help="Disable stratified CV for evaluation (use plain KFold).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logger = make_logger(level=logging.INFO)

    # Validate argument combinations
    if args.data and not args.target:
        print("Error: --target is required when --data is specified.", file=sys.stderr)
        sys.exit(1)
    if args.target and not args.data:
        print("Error: --target requires --data.", file=sys.stderr)
        sys.exit(1)
    if args.feature_types and not args.data:
        print("Error: --feature-types requires --data.", file=sys.stderr)
        sys.exit(1)

    fs_kwargs = dict(
        feature_selection=args.feature_selection,
        forward_selection=args.forward_selection,
        fs_tolerance=args.fs_tolerance,
        fs_min_features=args.fs_min_features,
        eval_strategy=args.eval_strategy,
        tuning_n_splits=args.tuning_folds,
        tuning_stratified=not args.no_tuning_stratify,
        eval_n_splits=args.eval_folds,
        eval_stratified=not args.no_eval_stratify,
    )

    # ── CSV dataset mode ─────────────────────────────────────────────
    if args.data:
        task = args.task if args.task != "all" else "clf"
        _run_from_csv(
            logger,
            data_path=args.data,
            target_column=args.target,
            feature_types_path=args.feature_types,
            task=task,
            **fs_kwargs,
        )
        return

    # ── Toy-dataset demo mode ────────────────────────────────────────
    if args.task in ("clf", "all"):
        _run_classification(logger, **fs_kwargs)

    if args.task in ("reg", "all"):
        _run_regression(logger, **fs_kwargs)


if __name__ == "__main__":
    main()