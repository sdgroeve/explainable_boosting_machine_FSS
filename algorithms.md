# Algorithmic Report: Explainable Boosting Machine with Feature Subset Selection

This document provides a detailed scientific description of the algorithms implemented in this codebase. The system trains, tunes, evaluates, and interprets **Explainable Boosting Machines (EBMs)** with optional **backward-elimination feature subset selection** and **feature-on-feature proxy testing**. All algorithms are described at a level of detail sufficient for full reproducibility.

---

## Table of Contents

1. [Explainable Boosting Machines (EBMs)](#1-explainable-boosting-machines-ebms)
2. [Hyperparameter Tuning](#2-hyperparameter-tuning)
3. [Evaluation Strategies](#3-evaluation-strategies)
4. [Backward Feature Elimination](#4-backward-feature-elimination)
5. [The Tune-Once Optimisation](#5-the-tune-once-optimisation)
6. [Feature-on-Feature Proxy Test](#6-feature-on-feature-proxy-test)
7. [Model Interpretability](#7-model-interpretability)
8. [t-SNE Visualisation of Local Explanations](#8-t-sne-visualisation-of-local-explanations)
9. [Performance Metrics](#9-performance-metrics)

---

## 1. Explainable Boosting Machines (EBMs)

### 1.1 Mathematical Foundation

An Explainable Boosting Machine is a **Generalised Additive Model with pairwise interactions** (GA²M). The prediction for a data point **x** = (x₁, x₂, …, xₚ) is:

$$
g\bigl(\mathbb{E}[y \mid \mathbf{x}]\bigr) = \beta_0 + \sum_{j=1}^{p} f_j(x_j) + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k)
$$

where:

- **g(·)** is a link function (identity for regression, logit for binary classification, softmax for multiclass).
- **β₀** is the intercept.
- **fⱼ(xⱼ)** is a **univariate shape function** (also called a partial response function) for feature *j*. Each shape function maps a single feature value to a real-valued contribution. These are learned non-parametrically, allowing the model to capture arbitrary non-linear relationships without requiring the user to specify the functional form.
- **f_{jk}(xⱼ, xₖ)** are **pairwise interaction shape functions** over selected pairs of features. These capture synergies or interactions that cannot be explained by the sum of individual effects alone.
- **𝓘** is the set of detected interactions (automatically selected by the FAST algorithm during training, or user-specified).

This formulation generalises linear models (where each fⱼ is constrained to be linear) to arbitrary non-linear relationships while retaining **full interpretability**: the contribution of each feature (and each interaction) to a prediction is directly readable from the shape functions.

### 1.2 Training Algorithm: Cyclic Gradient Boosting

EBMs are trained using a **cyclic boosted-trees** procedure. Unlike standard gradient boosting (e.g. XGBoost, LightGBM) which greedily selects the best feature at each boosting round, EBM trains each feature's shape function **in round-robin fashion**:

1. Initialise all shape functions fⱼ to zero and set the intercept β₀ to the target mean (regression) or log-odds (classification).
2. For each **boosting round** r = 1, …, R:
   - For each feature j = 1, …, p:
     - Compute the **pseudo-residuals** (negative gradient of the loss) given the current model predictions.
     - Fit a **small decision tree** (with at most `max_leaves` leaves) to predict the pseudo-residuals using only feature *j*.
     - Update fⱼ ← fⱼ + η · tree_j, where η is the learning rate.
3. After all univariate shape functions are learned, optionally detect and train pairwise interaction terms f_{jk} using the same cyclic boosting procedure.

Key design choices:

- **Round-robin cycling** ensures that all features are trained for the same number of rounds, preventing greedy dominance of high-signal features and reducing co-adaptation between feature estimates.
- **Small trees** (typically 2–8 leaves) keep each shape function interpretable as a step function.
- **Low learning rate** (0.01–0.2) and many rounds provide fine-grained, smooth estimates.
- **Bagging (outer bags)** reduces variance: multiple independent models are trained on bootstrap samples and their shape functions are averaged.

### 1.3 Feature Binning

Continuous features are discretised into at most `max_bins` equal-frequency bins (quantiles). This discretisation serves two purposes:

1. It bounds the computational cost independent of the number of unique feature values.
2. The resulting shape function is a step function whose breakpoints align with bin boundaries, making it easy to visualise and interpret.

Categorical (nominal) features are treated natively, with one bin per category.

### 1.4 Interaction Detection

EBMs use the **FAST** (Feature-interaction Automatic Selection Tool) algorithm to automatically detect which pairs of features have statistically significant interactions. The number of interactions to include is controlled by the `interactions` hyperparameter (0 disables interactions entirely). Interactions are trained after the main univariate shape functions are established.

### 1.5 EBM Variants

Two model classes are used depending on the task:

| Task | Class | Link Function | Loss |
|------|-------|--------------|------|
| Binary/Multiclass Classification | `ExplainableBoostingClassifier` | Logit / Softmax | Log-loss |
| Regression | `ExplainableBoostingRegressor` | Identity | Mean Squared Error |

Task type is **auto-detected** by examining the target variable (see §9.1), or can be forced by the user.

---

## 2. Hyperparameter Tuning

### 2.1 Search Strategy

Hyperparameter tuning is performed using scikit-learn's **RandomizedSearchCV** (default) or **GridSearchCV** (optional). The search is conducted using stratified k-fold cross-validation for classification (preserving class proportions in each fold) or standard k-fold for regression.

**Search procedure:**

1. An EBM estimator is created with the configured `random_state` and `n_jobs`.
2. A cross-validation splitter is created:
   - **StratifiedKFold** for classification (ensures each fold has approximately the same class ratio).
   - **KFold** for regression (random split without stratification).
3. Either `RandomizedSearchCV` (randomly sampling `n_iter` parameter combinations) or `GridSearchCV` (exhaustively evaluating all combinations) is fitted on the training data.
4. The best estimator (highest CV score) is extracted via `search.best_estimator_`.

### 2.2 Hyperparameter Search Space

The default search space explores the following EBM parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `learning_rate` | 20 values in [0.01, 0.20] (linearly spaced) | Step size for each boosting update |
| `max_bins` | {64, 128, 256} | Maximum number of bins for continuous features |
| `max_leaves` | {2, 3, 4, 5, 6, 8} | Maximum number of leaves per tree in each boosting step |
| `min_samples_leaf` | {1, 2, 5, 10, 20, 50} | Minimum samples in each leaf node |
| `max_rounds` | {200, 400, 800, 1200} | Number of boosting rounds |
| `interactions` | {0, 5, 10, 20} or {0} | Number of pairwise interactions to detect (0 = none) |
| `outer_bags` | {4, 8} | Number of bagging iterations |

### 2.3 Scoring Metrics

The scoring metric used during hyperparameter search depends on the task:

- **Binary classification:** `roc_auc` (Area Under the ROC Curve)
- **Multiclass classification:** `roc_auc_ovr` (One-vs-Rest ROC AUC, macro-averaged)
- **Regression:** `neg_root_mean_squared_error` (negative RMSE, since scikit-learn maximises scores)

---

## 3. Evaluation Strategies

Three evaluation strategies are supported, controlling how data is used for training and assessment:

### 3.1 Train/Test Split (`train_test`)

The standard holdout evaluation approach:

1. **Split** the data into training set (80% by default) and holdout test set (20%).
   - For classification, the split is **stratified** (preserving class proportions).
2. **Tune** hyperparameters using k-fold CV on the training set only.
3. **Evaluate** the best model on the holdout test set.
4. Report holdout metrics and generate all plots using test-set predictions.

This strategy provides unbiased performance estimates but reduces the amount of data available for training.

### 3.2 Cross-Validation Only (`cv_only`)

When data is limited, all samples can be used for both tuning and evaluation:

1. **Tune** hyperparameters using k-fold CV on all data (tuning folds, default 5).
2. **Generate out-of-fold (OOF) predictions** using a separate k-fold CV loop (evaluation folds, default 5):
   - For each fold, a clone of the best model is trained on the training portion and predictions are made on the held-out portion.
   - Fold scores are logged individually.
   - OOF predictions are assembled into a single vector covering all data points.
3. **Refit** the final model on all data for deployment.
4. Report aggregated CV metrics (mean ± std of fold scores).

The OOF prediction vector ensures that every sample is predicted by a model that **never saw it during training**, providing honest performance estimates without wasting data.

When `--save-predictions` is enabled under this strategy, local explanations are also collected out-of-fold: each fold's model produces local explanations for its held-out portion, and these are assembled into a full OOF explanation matrix.

### 3.3 Train Only (`train_only`)

A simplified mode for rapid prototyping:

1. **Tune** hyperparameters using k-fold CV on all data.
2. **Refit** the best model on all data.
3. **Evaluate** on the training data itself.

> **Warning:** Metrics computed on training data are **optimistic** and may overestimate generalisation performance. This mode should only be used when the goal is to inspect model behaviour on the training distribution, not to estimate test-time performance.

---

## 4. Backward Feature Elimination

### 4.1 Algorithm Overview

The feature selection algorithm implemented is **backward elimination** (also called backward stepwise selection). Starting from the full feature set, the algorithm iteratively removes the least important feature, retrains the model, and evaluates performance. Elimination continues until one of two stopping conditions is met.

### 4.2 Formal Algorithm

**Input:**
- Dataset (X, y) with p features
- Tolerance τ (e.g. 0.02 = 2%)
- Minimum features k_min
- Step percentage s (optional, default 0 = remove one at a time)

**Output:**
- Selected feature subset S* ⊂ {1, …, p}
- Elimination history

**Procedure:**

```
1.  S ← {1, 2, …, p}           (start with all features)
2.  Train EBM on X[S], compute CV score → score_baseline
3.  S* ← S, score* ← score_baseline
4.  WHILE |S| > k_min:
5.      Train EBM on X[S], extract term_importances()
6.      Sort features by importance (descending)
7.      IF s > 0:
8.          n_remove ← max(1, ⌊|S| × s⌋)
9.          n_remove ← min(n_remove, |S| - k_min)
10.     ELSE:
11.         n_remove ← 1
12.     Select the bottom n_remove features (least important) → R
13.     S_candidate ← S \ R
14.     Train EBM on X[S_candidate], compute CV score → score_new
15.     relative_drop ← (score_baseline - score_new) / |score_baseline|
16.     IF relative_drop > τ:
17.         STOP  (further removal degrades performance too much)
18.     S ← S_candidate
19.     S* ← S, score* ← score_new
20. RETURN S*, score*, history
```

### 4.3 Feature Importance Ranking

At each elimination step, features are ranked by their **term importances** as reported by `model.term_importances()`. This method returns the mean absolute contribution of each feature's shape function across all training samples. Only single-feature terms are considered; interaction terms (identified by the " & " separator or tuple format) are excluded from the ranking.

If `term_importances()` fails (e.g. due to a version mismatch), a fallback method uses `explain_global().data()` to retrieve importance scores.

### 4.4 Tolerance-Based Stopping

The stopping criterion uses **relative drop** rather than absolute score difference, making it robust to different scoring scales:

```
relative_drop = (score_baseline − score_current) / |score_baseline|
```

For maximised metrics (e.g. ROC AUC), a drop means the score decreased. For negated metrics (e.g. `neg_root_mean_squared_error`), a "drop" means the score became more negative. The formula handles both cases correctly.

Elimination stops when `relative_drop > τ`, and the **previous** feature set (before the drop exceeded tolerance) is returned as the selected subset.

### 4.5 Batch Feature Removal

When `step_percent > 0`, multiple features are removed at each step. This accelerates feature selection for high-dimensional datasets:

- `step_percent = 0.1` removes 10% of remaining features at each step.
- The number of features removed is always at least 1 and never reduces the set below `min_features`.
- Removed features are listed in the elimination history CSV.

### 4.6 Output Artifacts

The feature selection process produces:
- A `FeatureSelectionResult` object containing the selected features, scores, and step-by-step history.
- A `feature_selection_summary.csv` file with the full elimination history.
- An elimination curve plot in the HTML report showing CV score vs. number of features, with a shaded tolerance band.

---

## 5. The Tune-Once Optimisation

### 5.1 Motivation

By default, every call to `_fit_and_score` during feature elimination runs a full hyperparameter search (RandomizedSearchCV or GridSearchCV). For p features and O(p) elimination steps, this results in O(p × n_iter × k) model fits, which can be prohibitively slow.

### 5.2 Algorithm

When `tune_once` is enabled:

1. **Once, before elimination begins:** Run a full hyperparameter search on the complete feature set. Record the best parameters as `fixed_params`.
2. **At each elimination step:** Instead of searching, create an EBM with `fixed_params`, evaluate it via `cross_val_score`, and fit it on the current feature subset.

This reduces the cost of each elimination step from O(n_iter × k) fits to O(k) fits (just the cross-validation evaluation), a speedup factor of approximately n_iter.

### 5.3 Trade-off

The assumption is that optimal hyperparameters do not change significantly as features are removed. This is reasonable when:
- Features are removed one or a few at a time.
- The removed features are the least important ones.
- The hyperparameter space is not strongly coupled to the number of features.

For datasets where feature removal causes large structural changes in the learning problem, the default (re-tune at each step) may yield better results.

---

## 6. Feature-on-Feature Proxy Test

### 6.1 Purpose

After backward feature elimination selects a minimal feature subset S, a natural question is: **can the discarded features serve as "proxies" (stunt doubles) for the selected features?** If a selected feature can be accurately predicted from the discarded features alone, it suggests redundancy in the original feature set.

### 6.2 Algorithm

For each selected feature f ∈ S:

1. **Define target and predictors:**
   - Target: y = X[:, f] (the selected feature's values).
   - Predictors: X_proxy = X[:, D] where D = {all features} \ S (the discarded features).

2. **Determine task type:**
   - If the target feature has ≤ 20 unique values, or is string/categorical/boolean type → **classification**.
   - Otherwise → **regression**.

3. **Cross-validate a proxy model:**
   - Train an EBM (with `interactions=0` for speed) using 5-fold CV.
   - Classification scoring: ROC AUC (binary) or ROC AUC One-vs-Rest (multiclass).
   - Regression scoring: R².
   - Report the mean cross-validated score.

4. **Extract proxy feature importances:**
   - Fit a final EBM on all proxy data and retrieve `explain_global()` importances.
   - Report the top 5 most important proxy features.

### 6.3 Interpretation

| Score Range (Classification AUC) | Score Range (Regression R²) | Interpretation |
|---|---|---|
| ≥ 0.90 | ≥ 0.70 | **Highly Predictable** — the selected feature is largely redundant; discarded features capture its information. |
| 0.75 – 0.90 | 0.40 – 0.70 | **Moderately Predictable** — partial redundancy; some information is shared. |
| < 0.75 | < 0.40 | **Weakly Predictable** — the selected feature carries unique information not recoverable from the discarded features. |

### 6.4 Input: Ranked Feature List

The proxy test requires a ranking of features from the feature selection step. Features are ranked as follows:

1. Features that were **never removed** during backward elimination are ranked highest (they are the most important).
2. Features that were removed are ranked in reverse order of removal: the last to be removed is ranked higher than the first to be removed.
3. The top N features (user-specified, default 5) from this ranking are tested.

---

## 7. Model Interpretability

### 7.1 Global Explanations

Global explanations describe the model's behaviour across the entire dataset. The EBM provides two levels:

#### 7.1.1 Feature Importances

Feature importances are obtained via `model.term_importances()`, which returns the **mean absolute contribution** of each term (feature or interaction) to predictions across all samples. These importances reflect how much each feature, on average, influences the model's output.

The importance of feature *j* is defined as:

$$
\text{Importance}_j = \frac{1}{n} \sum_{i=1}^{n} |f_j(x_{ij})|
$$

#### 7.1.2 Shape Functions

Each feature's shape function fⱼ(xⱼ) is directly extracted from the trained model via `explain_global().data(index)`. The shape function is represented as:

- **Continuous features:** A step function defined by bin edges and corresponding contribution values. The contribution at each bin represents fⱼ at the midpoint of that bin.
- **Categorical features:** A mapping from each category to its contribution value.

Shape functions provide **complete transparency** into how each feature affects predictions:
- Positive contributions push the prediction toward higher values (or toward the positive class in classification).
- Negative contributions push toward lower values (or the negative class).
- The shape reveals non-linear relationships, thresholds, and saturation effects.

#### 7.1.3 Feature Interactions

Pairwise interaction shape functions f_{jk}(xⱼ, xₖ) are visualised as **heatmaps** where:
- The x-axis represents bins of feature *k*.
- The y-axis represents bins of feature *j*.
- The colour intensity represents the interaction contribution.

#### 7.1.4 Feature Density Plots

For each feature, a **kernel density estimate (KDE)** is plotted to show the distribution of feature values:
- **Classification:** Separate density curves for each class, enabling visual assessment of class separation.
- **Regression:** Density curves split by median target value (low vs. high), showing how feature distributions differ for low and high outcomes.

### 7.2 Local Explanations

Local explanations decompose individual predictions into per-feature contributions. For a specific data point **x**ᵢ, the prediction can be written as:

$$
\hat{y}_i = \beta_0 + \sum_{j=1}^{p} f_j(x_{ij}) + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_{ij}, x_{ik})
$$

Each term fⱼ(xᵢⱼ) is the local contribution of feature *j* for this data point. These are obtained via `model.explain_local(X, y)` and visualised as horizontal bar charts showing the top contributing features for each prediction.

### 7.3 Positive-Class Local Explanations

An optional report section generates local explanation plots for **every data point predicted as positive class** (label = 1). This is useful for:
- Understanding which features drive positive predictions.
- Auditing individual predictions (e.g. in medical or financial applications).
- Identifying common patterns among positive predictions.

---

## 8. t-SNE Visualisation of Local Explanations

### 8.1 Motivation

When `--save-predictions` is enabled, the system produces a `predictions.csv` file with per-feature local explanation values (columns `explain_<feature>`). Each row's explanation vector lives in a p-dimensional space. To visualise the structure of these explanation vectors, **t-SNE** (t-distributed Stochastic Neighbour Embedding) is used for dimensionality reduction.

### 8.2 Algorithm

t-SNE is a non-linear dimensionality reduction technique that preserves local neighbourhood structure. Given n data points with p-dimensional explanation vectors:

1. **Compute pairwise affinities in high-dimensional space:**
   For each pair of points (i, j), compute the conditional probability that point i would pick point j as its neighbour under a Gaussian centred at i:

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_k^2)}
$$

   The bandwidth σᵢ is chosen so that the entropy matches a user-specified **perplexity** (default: 30). The symmetrised affinity is p_{ij} = (p_{j|i} + p_{i|j}) / 2n.

2. **Initialise 2D embeddings** y₁, …, yₙ randomly.

3. **Compute pairwise affinities in low-dimensional space** using a Student-t distribution with one degree of freedom:

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

4. **Minimise the KL divergence** KL(P || Q) using gradient descent (1000 iterations).

### 8.3 Optional Preprocessing

Before applying t-SNE, explanation vectors can be normalised:

- **None** (default): Raw explanation values are used directly.
- **StandardScaler**: Each feature's explanations are centred to zero mean and unit variance.
- **MinMaxScaler**: Each feature's explanations are scaled to [0, 1].

### 8.4 Visualisation

The 2D t-SNE coordinates are plotted as scatter plots coloured by:
- **Classification tasks:** Discrete class labels using a qualitative colour palette (tab10/tab20).
- **Regression tasks:** Continuous target values using the viridis colourmap.

Clusters in the t-SNE plot indicate groups of samples that receive similar explanation patterns — i.e. they are "explained" in the same way by the model, even if their raw feature values differ.

---

## 9. Performance Metrics

### 9.1 Task Detection

The system automatically detects whether the target variable represents a classification or regression task:

1. If the target is string, categorical, or boolean → **classification**.
2. If the target is numeric with ≤ 20 unique values → **classification**.
3. Otherwise → **regression**.

The user can override this with `--task clf` or `--task reg`.

### 9.2 Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / N | Fraction of correct predictions |
| **F1 (macro)** | Mean of per-class F1 scores | Balanced measure of precision and recall |
| **ROC AUC** | ∫ TPR d(FPR) | Area under the ROC curve (binary) |
| **Average Precision** | ∫ Precision d(Recall) | Area under the precision-recall curve |
| **Brier Score** | (1/n) Σ(pᵢ − yᵢ)² | Calibration error (lower is better) |
| **ROC AUC OvR** | Macro-averaged one-vs-rest AUC | Multiclass generalisation of ROC AUC |

### 9.3 Regression Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **RMSE** | √[(1/n) Σ(yᵢ − ŷᵢ)²] | Root mean squared error |
| **MAE** | (1/n) Σ\|yᵢ − ŷᵢ\| | Mean absolute error |
| **R²** | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | Coefficient of determination |

### 9.4 Classification Diagnostic Plots

- **Confusion Matrix:** Grid showing TP, FP, TN, FN counts.
- **ROC Curve:** True Positive Rate vs. False Positive Rate.
- **Precision-Recall Curve:** Precision vs. Recall at varying thresholds.
- **Calibration Curve:** Predicted probability vs. observed fraction of positives (10 quantile bins).

### 9.5 Regression Diagnostic Plots

- **Residuals vs. Predicted:** Scatter plot to detect heteroscedasticity or non-linearity.
- **Residual Distribution:** Histogram to check normality of errors.
- **Predicted vs. True:** Scatter plot with identity line to visualise prediction accuracy.

---

## 10. Data Processing Pipeline

### 10.1 Input Validation

Before training, the input data undergoes rigorous validation:

1. Check for empty data (X or y has zero samples).
2. Check X and y have compatible lengths.
3. Check y for NaN, infinite, or missing values.
4. Check X for NaN values (warning, not error, since EBMs handle missing values natively).
5. Validate feature names match the number of columns.

### 10.2 Feature Types

Users can provide a feature-types CSV file specifying how each feature should be treated:

- **`continuous`**: Standard numeric binning.
- **`nominal`**: Categorical encoding (one bin per category).
- **Bracket-delimited lists** (e.g. `[1,2,3+]`): Custom ordinal bins.

If no feature types are provided, the EBM automatically infers types from the data.

### 10.3 Missing Target Handling

Rows with missing target values are automatically dropped before training, with a warning logged indicating the number and percentage of dropped rows.

---

## 11. System Architecture

The system is organised into the following modules:

| Module | Responsibility |
|--------|---------------|
| `run.py` | CLI entry point, argument parsing, demo datasets |
| `runner.py` | Core EBM training/tuning/evaluation orchestration |
| `feature_selection.py` | Backward elimination algorithm |
| `proxy_test.py` | Feature-on-feature redundancy analysis |
| `proxy_reporting.py` | HTML report for proxy test results |
| `config.py` | Configuration dataclasses |
| `metrics.py` | Classification and regression metrics computation |
| `plotting.py` | All matplotlib plotting functions |
| `html_templates.py` | HTML/CSS templates for reports |
| `html_reporting.py` | HTML report assembly |
| `utils.py` | Logging, validation, task detection |
| `visualize_explanations.py` | t-SNE visualisation of local explanations |

---

## References

1. Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2013). *Accurate Intelligible Models with Pairwise Interactions.* Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
2. Nori, H., Jenkins, S., Koch, P. & Caruana, R. (2019). *InterpretML: A Unified Framework for Machine Learning Interpretability.* arXiv:1909.09223.
3. van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE.* Journal of Machine Learning Research, 9, 2579–2605.
4. Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* Journal of Machine Learning Research, 13, 281–305.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
