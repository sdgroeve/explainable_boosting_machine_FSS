# EBM Runner

A comprehensive framework for training, tuning, evaluating, and interpreting **Explainable Boosting Machine (EBM)** models with automated HTML report generation, backward feature elimination, and feature-redundancy proxy testing.

> For a detailed scientific description of all algorithms, see [`algorithms.md`](algorithms.md).

## Features

- 🤖 **Automated Hyperparameter Tuning**: RandomizedSearchCV or GridSearchCV with sensible defaults
- 📊 **Comprehensive Metrics**: Classification (accuracy, F1, ROC-AUC, etc.) and regression (RMSE, MAE, R²)
- 🔍 **Advanced Model Interpretability**: 
    - Global feature importances
    - Shape functions for **all** features
    - Feature density plots (split by class or target median)
    - Automated detection and visualisation of feature interactions
    - Local explanations for individual samples
    - Optional positive-class local explanations for every predicted-positive sample
- 🧬 **Feature Selection**:
    - **Backward elimination** – iteratively removes the *least* important feature to find a minimal subset
    - **Tune-once** optimisation – tune hyperparameters once, reuse across all elimination steps
    - **Batch removal** – remove a percentage of features per step for faster selection
- 🔬 **Feature-on-Feature Proxy Test**: Analyse whether discarded features can predict selected features (redundancy analysis)
- 📈 **t-SNE Visualisation**: Project local explanation vectors into 2D to reveal explanation structure
- 📂 **CSV Dataset Support**: Load any CSV, specify a target column, and optionally supply feature-type definitions
- 🌐 **HTML Reports**: Single-file interactive reports with embedded plots, metrics, and full model interpretation
- 💾 **Predictions Export**: Save per-datapoint predictions with local explanation values to CSV

## Installation

### Prerequisites

- Python 3.8 or higher
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

Enable backward feature selection:

```bash
python run.py --data my_dataset.csv --target label --task clf \
    --feature-selection \
    --fs-tolerance 0.02 --fs-min-features 3
```

Use cross-validation only (no holdout split):

```bash
python run.py --data my_dataset.csv --target label --task clf \
    --eval-strategy cv_only --eval-folds 10
```

Save predictions with local explanations:

```bash
python run.py --data my_dataset.csv --target label --task clf \
    --save-predictions
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

# Train, tune, evaluate, and generate HTML report
artifacts = runner.fit_optimize_validate(
    X, y,
    feature_selection_result=fs_result,
)

print(f"Model saved to: {artifacts.model_path}")
print(f"Report saved to: {artifacts.report_html_path}")
print(f"Test metrics: {artifacts.test_result.metrics}")
```

## CLI Reference

### Core Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `all` | `clf`, `reg`, or `all`. When `--data` is not set, `all` runs both demo datasets. |
| `--data` | — | Path to a CSV file. Requires `--target`. |
| `--target` | — | Name of the target column in the CSV. |
| `--output-dir` | `./ebm_output` | Directory for all outputs (models, reports, CSVs). |
| `--feature-types` | — | Path to a two-column CSV (`feature`, `type`) with EBM feature-type definitions. |

### Feature Selection

| Argument | Default | Description |
|---|---|---|
| `--feature-selection` | off | Enable backward-elimination feature selection. |
| `--fs-tolerance` | `0.02` | Maximum relative CV-score drop allowed during elimination (0.02 = 2%). |
| `--fs-min-features` | `3` | Minimum number of features to keep. |
| `--fs-step-percent` | `0.0` | Percentage of features to remove per step (0 = one at a time). |
| `--tune-once` | off | Tune hyperparameters once on full features, reuse during elimination. |

### Evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval-strategy` | `cv_only` | `train_test`, `cv_only`, or `train_only` (see below). |
| `--tuning-folds` | `5` | Number of CV folds for hyperparameter tuning. |
| `--eval-folds` | `5` | Number of CV folds for `cv_only` evaluation. |
| `--no-tuning-stratify` | off | Disable stratified CV for tuning. |
| `--no-eval-stratify` | off | Disable stratified CV for evaluation. |
| `--ebm-n-jobs` | `-1` | Number of parallel jobs for EBM internal training. |

### Reporting & Export

| Argument | Default | Description |
|---|---|---|
| `--save-predictions` | off | Write `predictions.csv` with per-datapoint predictions and local explanations. |
| `--explain-positive-class` | off | Add local explanation plots for every positive-class prediction (classification only). |

### Proxy Test

| Argument | Default | Description |
|---|---|---|
| `--proxy-features-file` | — | Path to a feature selection summary CSV for proxy testing. |
| `--proxy-num-features` | `5` | Number of top features to analyse in the proxy test. |

## Evaluation Strategies

| Strategy | Behaviour | Report shows |
|---|---|---|
| `train_test` | Split → CV-tune on train → evaluate on holdout test set. | Holdout test metrics + validation plots |
| `cv_only` | CV-tune on all data; no holdout. Out-of-fold predictions used for plots. | CV-aggregate metrics + per-fold scores |
| `train_only` | CV-tune on all data, refit on all data. | Training-set metrics + overfitting warning |

Examples:

```bash
# Default train/test split
python run.py --task clf --eval-strategy train_test

# 10-fold CV evaluation, 3-fold tuning
python run.py --task clf --eval-strategy cv_only --eval-folds 10 --tuning-folds 3

# Train-only: tune with 5-fold CV, refit on all data
python run.py --task clf --eval-strategy train_only
```

## Visualising Local Explanations

After generating predictions with `--save-predictions`, use the standalone visualisation script to project local explanation vectors into 2D with t-SNE:

```bash
python visualize_explanations.py --predictions ./ebm_output/predictions.csv
```

Options:

| Argument | Default | Description |
|---|---|---|
| `--predictions` | (required) | Path to `predictions.csv` with `explain_*` columns. |
| `--normalize` | `none` | Normalisation: `none`, `standard`, or `minmax`. |
| `--perplexity` | `30` | t-SNE perplexity parameter. |
| `--label-column` | `y_true` | Column to use for colouring points. |
| `--output-dir` | same as input | Directory for the HTML report and updated CSV. |

## Code Structure

```
├── run.py                          # CLI entry point
├── visualize_explanations.py       # t-SNE visualisation of local explanations
├── algorithms.md                   # Detailed scientific report of all algorithms
└── ebm_runner/                     # Main package
    ├── __init__.py                 # Public exports
    ├── config.py                   # Configuration dataclasses
    ├── utils.py                    # Logging, validation, helpers
    ├── runner.py                   # EBMRunner orchestrator
    ├── feature_selection.py        # Backward feature elimination
    ├── metrics.py                  # Metric computation
    ├── plotting.py                 # All matplotlib plotting functions
    ├── html_templates.py           # HTML/CSS template strings
    ├── html_reporting.py           # HTML report assembly
    ├── proxy_test.py               # Feature-on-feature proxy testing
    └── proxy_reporting.py          # Proxy test HTML report
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

                      ┌─────────────────┐
                      │  Proxy Test      │
                      │  (proxy_test.py) │
                      └────────┬────────┘
                               ▼
                      ProxyReportGenerator
                      (proxy_reporting.py)
```

### Module Details

#### `run.py` – CLI entry point

Defines the argument parser and run functions:

- **`_run_from_csv`** – loads a user-supplied CSV, optionally loads a feature-type definition file, runs feature selection, trains the model, and generates the report.
- **`_run_classification` / `_run_regression`** – convenience wrappers that use scikit-learn toy datasets for quick demos.
- **`_run_proxy_test`** – loads a feature ranking, selects top features, and runs the proxy test.

#### `ebm_runner/runner.py` – `EBMRunner`

The central orchestrator. Key methods:

| Method | Purpose |
|---|---|
| `fit_optimize_validate` | End-to-end pipeline: prepares data → runs hyper-parameter search → evaluates → generates HTML report. |
| `select_features` | Convenience wrapper that creates a `FeatureSelector` and runs backward elimination. |
| `_run_train_test` | Train/test split strategy implementation. |
| `_run_cv_only` | Cross-validation only strategy with OOF predictions. |
| `_run_train_only` | Train-only strategy (no holdout). |
| `_make_estimator` | Instantiates `ExplainableBoostingClassifier` or `ExplainableBoostingRegressor`. |

#### `ebm_runner/feature_selection.py` – `FeatureSelector`

Implements backward-elimination feature selection:

1. Train an EBM on all features and record the baseline CV score.
2. At each step, rank features by `term_importances()`.
3. Remove the least important feature(s).
4. Retrain, compare the new CV score to the baseline.
5. Stop when the relative score drop exceeds `tolerance` or `min_features` is reached.

Produces a `FeatureSelectionResult` containing:
- `selected_features` – features remaining after elimination
- `baseline_score` / `final_score` – CV scores before and after
- `tolerance` – configuration used
- `history` – a DataFrame with one row per step (step number, removed feature, CV score, delta)

#### `ebm_runner/proxy_test.py` – `ProxyTestAnalyzer`

For each selected feature, trains an EBM on the discarded features to predict it. Reports the cross-validated predictability score and top proxy features. A high score indicates redundancy.

## Output Structure

After running, the output directory contains:

```
output_dir/
├── best_ebm_model.joblib          # Trained model
├── cv_results.csv                  # Cross-validation results
├── ebm_report.html                 # Comprehensive HTML report
├── predictions.csv                 # (if --save-predictions) Per-datapoint predictions + explanations
├── feature_selection_summary.csv   # (if --feature-selection) Elimination history
├── proxy_report.html               # (if proxy test was run) Proxy test results
└── plots/                          # Plot images directory
```

## HTML Report Contents

The generated HTML report includes:

1. **Run Summary** – task type, data sizes, feature count, evaluation strategy
2. **Performance Metrics** – detailed metrics table
3. **Validation Plots** – confusion matrix / ROC / PR / calibration (classification) or residual diagnostics (regression)
4. **Global Feature Importance** – ranking of all features
5. **Individual Feature Analysis** – shape functions + density plots for every feature
6. **Feature Interactions** – heatmaps for all detected interactions
7. **Local Explanations** – per-sample feature contribution bar charts
8. **Hyperparameter Tuning** – full CV results table with all evaluated candidates
9. **Feature Selection** *(optional)* – summary, history table, elimination curve
10. **Positive-Class Explanations** *(optional)* – local explanations for all positive-class predictions

All plots are embedded as base64 PNG images, so the report is a single portable HTML file.

## Troubleshooting

**`ImportError: Could not import interpret.glassbox`**

Install the interpret package:
```bash
pip install interpret
```

---

**Memory errors during hyperparameter search**

Reduce parallelisation or search space:
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

### v1.3.0 (Current)
- Feature-on-feature proxy testing (`--proxy-features-file`)
- t-SNE visualisation of local explanation vectors (`visualize_explanations.py`)
- Predictions CSV export with per-feature local explanations (`--save-predictions`)
- Positive-class local explanations (`--explain-positive-class`)
- Tune-once optimisation for feature selection (`--tune-once`)
- Batch feature removal (`--fs-step-percent`)
- Cross-validation only and train-only evaluation strategies
- Feature density plots in reports
- Removed legacy PDF reporting module

### v1.2.0
- CSV dataset loading via `--data` / `--target` CLI arguments
- Feature-type definition file support (`--feature-types`)
- Backward-elimination feature selection (`--feature-selection`)
- Robust handling of non-numeric targets and missing values

### v1.1.0
- Switched from PDF to interactive HTML reports
- Added automated feature interaction visualisation
- Comprehensive shape plots for all features
- Embedded plots for single-file portability

### v1.0.0
- Initial modular release
- Automated hyperparameter tuning
- Global and local interpretability
