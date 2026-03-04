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
    output_dir: str = "./ebm_output",
    feature_types_path: Optional[str] = None,
    task: str = "clf",
    feature_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    fs_step_percent: float = 0.0,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
    tune_once: bool = False,
    explain_positive_class: bool = False,
    save_predictions: bool = False,
    ebm_n_jobs: int = -1,
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
        output_dir=output_dir,
        force_task=force_task,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        tune_once=tune_once,
        explain_positive_class=explain_positive_class,
        save_predictions=save_predictions,
        ebm_n_jobs=ebm_n_jobs,
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
            step_percent=fs_step_percent,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_report.html",
        feature_selection_result=fs_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


def _run_classification(
    logger: logging.Logger,
    *,
    output_dir: str = "./ebm_breast_cancer",
    feature_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    fs_step_percent: float = 0.0,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
    tune_once: bool = False,
    explain_positive_class: bool = False,
    save_predictions: bool = False,
    ebm_n_jobs: int = -1,
) -> None:
    """Run the breast-cancer classification demo."""
    logger.info("=" * 60)
    logger.info("Classification Example (Breast Cancer)")
    logger.info("=" * 60)

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    runner = EBMRunner(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        tune_once=tune_once,
        explain_positive_class=explain_positive_class,
        save_predictions=save_predictions,
        ebm_n_jobs=ebm_n_jobs,
        logger=logger,
    )

    # Optional backward feature selection
    fs_result = None
    if feature_selection:
        fs_result = runner.select_features(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
            step_percent=fs_step_percent,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_classification_report.html",
        feature_selection_result=fs_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


def _run_regression(
    logger: logging.Logger,
    *,
    output_dir: str = "./ebm_diabetes",
    feature_selection: bool = False,
    fs_tolerance: float = 0.02,
    fs_min_features: int = 3,
    fs_step_percent: float = 0.0,
    eval_strategy: str = "train_test",
    tuning_n_splits: int = 5,
    tuning_stratified: bool = True,
    eval_n_splits: int = 5,
    eval_stratified: bool = True,
    tune_once: bool = False,
    explain_positive_class: bool = False,
    save_predictions: bool = False,
    ebm_n_jobs: int = -1,
) -> None:
    """Run the diabetes regression demo."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Regression Example (Diabetes)")
    logger.info("=" * 60)

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    runner = EBMRunner(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        tuning_n_splits=tuning_n_splits,
        tuning_stratified=tuning_stratified,
        eval_n_splits=eval_n_splits,
        eval_stratified=eval_stratified,
        tune_once=tune_once,
        explain_positive_class=explain_positive_class,
        save_predictions=save_predictions,
        ebm_n_jobs=ebm_n_jobs,
        logger=logger,
    )

    # Optional backward feature selection
    fs_result = None
    if feature_selection:
        fs_result = runner.select_features(
            X, y,
            tolerance=fs_tolerance,
            min_features=fs_min_features,
            step_percent=fs_step_percent,
        )
        logger.info(
            f"Backward: selected {len(fs_result.selected_features)}/{X.shape[1]} features: "
            f"{fs_result.selected_features}"
        )

    artifacts = runner.fit_optimize_validate(
        X, y,
        report_name="ebm_regression_report.html",
        feature_selection_result=fs_result,
    )

    logger.info(f"HTML report: {artifacts.report_html_path}")
    logger.info(f"Model:       {artifacts.model_path}")
    logger.info(f"Test metrics: {artifacts.test_result.metrics}")


def _run_proxy_test(
    logger: logging.Logger,
    data_path: str,
    target_column: str,
    proxy_features_file: str,
    proxy_num_features: int,
    output_dir: str = "./ebm_output",
    n_jobs: int = -1,
) -> None:
    from ebm_runner.proxy_test import ProxyTestAnalyzer
    from ebm_runner.proxy_reporting import ProxyReportGenerator
    import pandas as pd

    logger.info("=" * 60)
    logger.info("Proxy Test Mode")
    logger.info("=" * 60)

    try:
        data_df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to load dataset {data_path}: {e}")
        return

    try:
        proxy_df = pd.read_csv(proxy_features_file)
    except Exception as e:
        logger.error(f"Failed to load proxy features file {proxy_features_file}: {e}")
        return

    if target_column in data_df.columns:
        data_df = data_df.drop(columns=[target_column])

    all_features = data_df.columns.tolist()

    if "removed_feature" in proxy_df.columns:
        removed_in_order = proxy_df["removed_feature"].dropna().tolist()
        removed_in_order = [f.strip() for item in removed_in_order for f in str(item).split(",")]
    else:
        # Fallback if it's just a list of features in the first column
        removed_in_order = proxy_df.iloc[:, 0].dropna().tolist()
        removed_in_order = [f.strip() for item in removed_in_order for f in str(item).split(",")]

    # The features NOT removed are the most important.
    never_removed = [f for f in all_features if f not in removed_in_order]
    
    # Rank: never removed (best), then last removed, ..., first removed (worst)
    ranked_features = never_removed + list(reversed(removed_in_order))
    
    # Keep only those present in dataset
    ranked_features = [f for f in ranked_features if f in data_df.columns]

    # Select top N
    selected_features = ranked_features[:proxy_num_features]
    
    logger.info(f"Selected Top {proxy_num_features} features to test: {selected_features}")

    analyzer = ProxyTestAnalyzer(logger=logger, n_jobs=n_jobs)
    results = analyzer.analyze(data_df, selected_features)

    reporter = ProxyReportGenerator(output_dir=output_dir, logger=logger)
    reporter.generate_report(results)


# ── CLI ──────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EBM model training with optional feature selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ebm_output",
        help="Directory to save EBM outputs (models, reports, etc.). If it does not exist, it will be created.",
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
            "Path to a CSV file with columns 'feature' and 'type'. "
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
        "--proxy-features-file",
        type=str,
        default=None,
        help="Path to the feature selection summary CSV for Proxy testing.",
    )
    parser.add_argument(
        "--proxy-num-features",
        type=int,
        default=5,
        help="Number of most important features to analyze in the Proxy Test.",
    )
    parser.add_argument(
        "--fs-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative CV-score drop allowed during elimination (default: 0.02 = 2%%).",
    )
    parser.add_argument(
        "--fs-step-percent",
        type=float,
        default=0.0,
        help="Percentage of current features to remove at each step (e.g., 0.1 for 10%%). Default 0 removes 1 by 1.",
    )
    parser.add_argument(
        "--fs-min-features",
        type=int,
        default=3,
        help="Minimum number of features to keep (default: 3).",
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
        "--tune-once",
        action="store_true",
        default=_DEFAULTS.tune_once,
        help="Tune hyperparameters once on the full dataset, then reuse for feature selection steps.",
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
    parser.add_argument(
        "--explain-positive-class",
        action="store_true",
        default=_DEFAULTS.explain_positive_class,
        help=(
            "(Classification only) Add a dedicated report section with a local explanation "
            "plot for every datapoint predicted in the positive class."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=_DEFAULTS.save_predictions,
        help="Write predictions.csv to the output directory with per-datapoint predictions in original dataset order.",
    )
    parser.add_argument(
        "--ebm-n-jobs",
        type=int,
        default=_DEFAULTS.ebm_n_jobs,
        help=f"Number of parallel jobs for EBM internal training (default: {_DEFAULTS.ebm_n_jobs}).",
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
        output_dir=args.output_dir,
        feature_selection=args.feature_selection,
        fs_tolerance=args.fs_tolerance,
        fs_min_features=args.fs_min_features,
        fs_step_percent=args.fs_step_percent,
        eval_strategy=args.eval_strategy,
        tuning_n_splits=args.tuning_folds,
        tuning_stratified=not args.no_tuning_stratify,
        eval_n_splits=args.eval_folds,
        eval_stratified=not args.no_eval_stratify,
        tune_once=args.tune_once,
        explain_positive_class=args.explain_positive_class,
        save_predictions=args.save_predictions,
        ebm_n_jobs=args.ebm_n_jobs,
    )

    # ── CSV dataset mode ─────────────────────────────────────────────
    if args.data:
        if args.proxy_features_file:
            _run_proxy_test(
                logger,
                data_path=args.data,
                target_column=args.target,
                proxy_features_file=args.proxy_features_file,
                proxy_num_features=args.proxy_num_features,
                output_dir=args.output_dir,
                n_jobs=args.ebm_n_jobs or -1,
            )
            return

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