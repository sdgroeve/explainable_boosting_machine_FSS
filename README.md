# EBM Runner

A comprehensive framework for training, tuning, evaluating, and interpreting **Explainable Boosting Machine (EBM)** models with automated HTML report generation.

## Features

- 🤖 **Automated Hyperparameter Tuning**: RandomizedSearchCV or GridSearchCV with sensible defaults
- 📊 **Comprehensive Metrics**: Classification (accuracy, F1, ROC-AUC, etc.) and regression (RMSE, MAE, R²)
- 🔍 **Advanced Model Interpretability**: 
    - Global feature importances
    - Shape functions for **all** features
    - Automated detection and visualization of feature interactions
    - Local explanations for individual samples
- 🧬 **Feature Selection**:
    - **Backward elimination** – iteratively removes the *least* important feature to find a minimal subset
    - **Forward elimination** – iteratively removes the *most* important feature to reveal redundancy
    - Both modes can run in a single invocation and appear side-by-side in the HTML report
- 📂 **CSV Dataset Support**: Load any CSV, specify a target column, and optionally supply feature-type definitions
- 🌐 **HTML Reports**: Single-file interactive reports with embedded plots, metrics, and full model interpretation
- 🧩 **Modular Design**: Clean, maintainable code structure with separate modules for each concern
- ✅ **Input Validation**: Robust error checking for numeric and non-numeric targets, missing values, etc.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` – Numerical computing
- `pandas` – Data manipulation
- `matplotlib` – Plotting
- `scikit-learn` – Machine learning utilities
- `joblib` – Model persistence
- `interpret` – EBM models

## Quick Start

### Command Line

Run both built-in demo datasets (breast cancer classification + diabetes regression):

```bash
python run.py
```

Run with your own CSV data:

```bash
python run.py --data my_dataset.csv --target label --task clf
```

Enable both feature-selection modes:

```bash
python run.py --data my_dataset.csv --target label --task clf \
    --feature-selection --forward-selection \
    --fs-tolerance 0.02 --fs-min-features 3
```

### Python API

```python
from ebm_runner import EBMRunner
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

runner = EBMRunner(
    output_dir="./my_classification_run",
    n_iter=30,
    n_splits=5,
    enable_interactions=True,
)

# Optional: backward feature selection
fs_result = runner.select_features(X, y, tolerance=0.02, min_features=3)

# Optional: forward feature selection (redundancy analysis)
fwd_result = runner.select_features_forward(X, y, tolerance=0.02, min_features=3)

# Train, tune, evaluate, and generate HTML report
artifacts = runner.fit_optimize_validate(
    X, y,
    feature_selection_result=fs_result,
    forward_selection_result=fwd_result,
)

print(f"Model saved to: {artifacts.model_path}")
print(f"Report saved to: {artifacts.report_html_path}")
print(f"Test metrics: {artifacts.test_result.metrics}")
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--task` | `all` | `clf`, `reg`, or `all`. When `--data` is not set, `all` runs both demo datasets. |
| `--data` | — | Path to a CSV file. Requires `--target`. |
| `--target` | — | Name of the target column in the CSV. |
| `--feature-types` | — | Path to a two-column CSV (`feature`, `type`) with EBM feature-type definitions. |
| `--feature-selection` | off | Enable backward-elimination feature selection. |
| `--forward-selection` | off | Enable forward-elimination (redundancy analysis) feature selection. |
| `--fs-tolerance` | `0.02` | Maximum relative CV-score drop allowed during elimination (0.02 = 2%). |
| `--fs-min-features` | `3` | Minimum number of features to keep. |
| `--eval-strategy` | `train_test` | `train_test`, `cv_only`, or `train_only` (see below). |
| `--tuning-folds` | `5` | Number of CV folds for hyperparameter tuning. |
| `--eval-folds` | `5` | Number of CV folds for cv_only evaluation. |
| `--no-tuning-stratify` | off | Disable stratified CV for tuning. |
| `--no-eval-stratify` | off | Disable stratified CV for evaluation. |

## Evaluation Strategies

| Strategy | Behaviour | Report shows |
|---|---|---|
| `train_test` | Split → CV-tune on train → evaluate on holdout test set. | Holdout test metrics + validation plots |
| `cv_only` | CV-tune on all data; no holdout. Out-of-fold predictions used for plots. | CV-aggregate metrics + per-fold scores |
| `train_only` | CV-tune on all data, refit on all data. | Training-set metrics + overfitting warning |

Examples:

```bash
# Default train/test split
python run.py --task clf

# 10-fold CV evaluation, 3-fold tuning
python run.py --task clf --eval-strategy cv_only --eval-folds 10 --tuning-folds 3

# CV without stratification on evaluation folds
python run.py --task clf --eval-strategy cv_only --no-eval-stratify

# Train-only: tune with 5-fold CV, refit on all data
python run.py --task clf --eval-strategy train_only
```

## Code Structure

```
ML/
├── run.py                        # CLI entry point
└── ebm_runner/                   # Main package
    ├── __init__.py               # Public exports
    ├── config.py                 # Configuration dataclasses
    ├── utils.py                  # Logging, validation, helpers
    ├── runner.py                 # EBMRunner orchestrator
    ├── feature_selection.py      # Feature elimination logic
    ├── metrics.py                # Metric computation
    ├── plotting.py               # All matplotlib plotting
    ├── html_templates.py         # HTML/CSS template strings
    ├── html_reporting.py         # Report assembly
    └── reporting.py              # Legacy PDF reporting (unused)
```

### How the pieces fit together

```
                   run.py  (CLI)
                      │
                      ▼
                  EBMRunner  (runner.py)
                 ╱    │    ╲
                ╱     │     ╲
               ▼      ▼      ▼
  FeatureSelector  Hyper-param   HTMLReportGenerator
  (feature_        search /CV    (html_reporting.py)
   selection.py)   (sklearn)         │
       │                             ▼
       │                    HTML templates + plots
       │                    (html_templates.py,
       ▼                     plotting.py)
  FeatureSelectionResult
```

### Module details

#### `run.py` – CLI entry point

Defines the argument parser and three "run" functions:

- **`_run_from_csv`** – loads a user-supplied CSV, optionally loads a feature-type definition file, runs feature selection(s), trains the model, and generates the report.
- **`_run_classification` / `_run_regression`** – convenience wrappers that use scikit-learn toy datasets for quick demos.

Each function creates an `EBMRunner`, optionally calls `select_features()` and/or `select_features_forward()`, then calls `fit_optimize_validate()`.

#### `ebm_runner/runner.py` – `EBMRunner`

The central orchestrator. Key concepts:

| Method | Purpose |
|---|---|
| `__init__` | Stores all configuration (output dir, CV folds, tuning iterations, feature names/types, etc.). |
| `fit_optimize_validate` | End-to-end pipeline: prepares data → runs hyper-parameter search → evaluates on the test set → generates the HTML report. Accepts optional `feature_selection_result` and `forward_selection_result` to include in the report. |
| `select_features` | Convenience wrapper that creates a `FeatureSelector(direction="backward")` and runs it. |
| `select_features_forward` | Same, but with `direction="forward"`. |
| `_make_estimator` | Instantiates `ExplainableBoostingClassifier` or `ExplainableBoostingRegressor`, forwarding any user-supplied `feature_names` / `feature_types`. |
| `_prepare_data` | Coerces raw arrays / DataFrames into a standard `(DataFrame, ndarray, feature_names)` triple. |

#### `ebm_runner/feature_selection.py` – `FeatureSelector`

Implements iterative feature elimination:

1. Train an EBM on all features and record the baseline CV score.
2. At each step, rank features by `term_importances()`.
3. Remove the **least** important (backward) or **most** important (forward) feature.
4. Retrain, compare the new CV score to the baseline.
5. Stop when the relative score drop exceeds `tolerance` or `min_features` is reached.

The `direction` parameter (`"backward"` or `"forward"`) controls which feature is removed. Both modes produce a `FeatureSelectionResult` dataclass containing:

- `selected_features` – features remaining after elimination
- `baseline_score` / `final_score` – CV scores before and after
- `tolerance` / `direction` – configuration used
- `history` – a DataFrame with one row per step (step number, removed feature, CV score, delta)

#### `ebm_runner/html_reporting.py` / `html_templates.py` – HTML reports

`HTMLReportGenerator.generate_report()` assembles the final HTML file section by section:

1. Title & run summary
2. Table of contents (dynamically adjusts based on which feature-selection modes were used)
3. Performance metrics & validation plots
4. Global feature importance
5. Individual feature shape functions
6. Feature interactions
7. Local explanations
8. *(Optional)* Backward elimination results – summary, history table, elimination curve
9. *(Optional)* Forward elimination results – same layout, different heading and description

All plots are embedded as base64 PNG images, so the report is a single portable HTML file.

#### `ebm_runner/config.py`

Dataclasses for configuration:

- **`RunConfig`** – training parameters (splits, iterations, etc.)
- **`PlotConfig`** – plotting defaults (figure sizes, DPI)
- **`ReportConfig`** – report-specific settings (top-N features, local samples count)

#### `ebm_runner/utils.py`

Shared helpers:

- **`make_logger`** – creates a configured logger
- **`validate_input_data`** – checks for NaN, inf, shape mismatches; handles both numeric and non-numeric targets safely
- **`is_classification_target`** – auto-detects task type from the target values
- **`ensure_output_dir`** – creates the output directory tree

#### `ebm_runner/metrics.py`

- **`compute_classification_metrics`** – accuracy, precision, recall, F1, ROC-AUC, log-loss
- **`compute_regression_metrics`** – RMSE, MAE, R², explained variance
- **`ValidationResult`** dataclass returned by the runner

#### `ebm_runner/plotting.py`

Pure matplotlib plotting functions, each returning a `Figure`:

- `plot_feature_importances` – horizontal bar chart
- `plot_shape_for_feature` – EBM shape function
- `plot_interaction_shape` – 2D interaction heatmap
- `plot_local_explanation_bar` – per-sample feature contributions
- `plot_classification_curves` – ROC, PR, calibration
- `plot_regression_diagnostics` – residuals, distribution, predicted-vs-true
- `plot_confusion_matrix`
- `plot_elimination_curve` – CV score vs. number of features (used by both elimination modes)
- `fig_to_base64` – serializes any figure to a base64 string for HTML embedding

## EBMRunner Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | str | `"./ebm_output"` | Directory for outputs |
| `random_state` | int | `42` | Random seed for reproducibility |
| `force_task` | str \| None | `None` | Force `'classification'` or `'regression'` |
| `test_size` | float | `0.2` | Fraction of data for holdout test |
| `val_size` | float | `0.0` | Fraction for validation (0 = use CV only) |
| `n_splits` | int | `5` | Number of CV folds |
| `n_iter` | int | `40` | Iterations for RandomizedSearchCV |
| `use_grid` | bool | `False` | Use GridSearchCV instead |
| `n_jobs` | int | `-1` | Parallel jobs for search (-1 = all cores) |
| `ebm_n_jobs` | int | `1` | Parallel jobs for EBM training |
| `enable_interactions` | bool | `True` | Enable interaction terms in search |
| `eval_strategy` | str | `"train_test"` | `"train_test"`, `"cv_only"`, or `"train_only"` |
| `tuning_n_splits` | int | `5` | CV folds for hyperparameter tuning |
| `tuning_stratified` | bool | `True` | Stratify tuning folds for classification |
| `eval_n_splits` | int | `5` | CV folds for cv_only evaluation |
| `eval_stratified` | bool | `True` | Stratify eval folds for classification |
| `feature_names` | list \| None | `None` | Feature names passed to the EBM |
| `feature_types` | list \| None | `None` | Feature types passed to the EBM |
| `n_local_samples_in_report` | int | `3` | Local explanations in report |

## Output Structure

After running, the output directory contains:

```
output_dir/
├── best_ebm_model.joblib          # Trained model
├── cv_results.csv                  # Cross-validation results
├── ebm_report.html                 # Comprehensive HTML report
└── plots/                          # Generated plot images (also embedded in HTML)
    ├── feature_importances.png
    ├── shape_<feature>.png
    ├── local_0.png
    ├── confusion_matrix.png        # (classification)
    ├── class_curve_*.png           # (classification)
    └── reg_diag_*.png              # (regression)
```

## HTML Report Contents

The generated HTML report includes:

1. **Run Summary** – task type, data sizes, feature count
2. **Performance Metrics** – detailed test-set metrics
3. **Validation Plots** – confusion matrix / ROC / PR (classification) or residual diagnostics (regression)
4. **Global Feature Importance** – ranking of all features and interactions
5. **Individual Feature Analysis** – shape functions for every feature
6. **Feature Interactions** – plots for all detected interactions
7. **Local Interpretation** – per-sample feature contributions
8. **Feature Subset Selection (Backward)** – *(when `--feature-selection` is enabled)* summary, history table, elimination curve
9. **Feature Redundancy Analysis (Forward)** – *(when `--forward-selection` is enabled)* same layout, revealing how much information top features share with others

## Troubleshooting

**`ImportError: Could not import interpret.glassbox`**

Install the interpret package:
```bash
pip install interpret
```

---

**Memory errors during hyperparameter search**

Reduce parallelization or search space:
```python
runner = EBMRunner(n_jobs=1, n_iter=20, ebm_n_jobs=1)
```

---

**`ValueError: X and y have incompatible lengths`**

Ensure your feature matrix and target have the same number of samples:
```python
print(f"X shape: {X.shape}, y shape: {y.shape}")
assert len(X) == len(y)
```

## License

This code is provided as-is for educational and research purposes.

## Acknowledgments

- Built on Microsoft's [InterpretML](https://github.com/interpretml/interpret) library
- Uses scikit-learn for model selection and metrics

## Version History

### v1.2.0 (Current)
- CSV dataset loading via `--data` / `--target` CLI arguments
- Feature-type definition file support (`--feature-types`)
- Backward-elimination feature selection (`--feature-selection`)
- Forward-elimination / redundancy analysis (`--forward-selection`)
- Robust handling of non-numeric targets and missing values

### v1.1.0
- Switched from PDF to interactive HTML reports
- Added automated feature interaction visualization
- Comprehensive shape plots for all features
- Embedded plots for single-file portability

### v1.0.0
- Initial modular release
- Legacy PDF reporting
- Automated hyperparameter tuning
- Global and local interpretability
