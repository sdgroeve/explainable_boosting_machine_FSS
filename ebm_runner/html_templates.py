"""
HTML templates and styling for EBM report generation.

This module provides HTML templates and CSS styling for creating
comprehensive, browser-based reports with embedded visualizations.
"""

# Base HTML template with CSS styling
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EBM Model Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #555;
            font-size: 1.3em;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        .summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .summary p {{
            margin: 8px 0;
            font-size: 1.1em;
        }}
        
        .summary strong {{
            color: #2c3e50;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .metric-value {{
            font-weight: 600;
            color: #27ae60;
        }}
        
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background: white;
        }}
        
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .plot-grid-item {{
            text-align: center;
        }}
        
        .plot-grid-item img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        
        .feature-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        
        .interaction-section {{
            margin: 30px 0;
            padding: 20px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }}
        
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 20px;
        }}
        
        .toc li {{
            margin: 8px 0;
        }}
        
        .toc a {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #0c5460;
            color: #0c5460;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #856404;
            color: #856404;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .importance-bar {{
            display: inline-block;
            height: 20px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 3px;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

# Section templates
TITLE_SECTION = """
<h1>Explainable Boosting Machine (EBM) - Model Report</h1>
<p class="timestamp">Generated: {timestamp}</p>
"""

SUMMARY_SECTION = """
<div class="summary">
    <h2>Run Summary</h2>
    <p><strong>Task Type:</strong> {task_type}</p>
    <p><strong>Training Samples:</strong> {n_train}</p>
    <p><strong>Test Samples:</strong> {n_test}</p>
    <p><strong>Number of Features:</strong> {n_features}</p>
    <p><strong>Model Type:</strong> Explainable Boosting Machine (EBM)</p>
    <p><strong>Evaluation Strategy:</strong> {eval_strategy_label}</p>
{eval_strategy_note}
</div>
"""

TOC_SECTION = """
<div class="toc">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#performance">1. Prediction Performance</a></li>
        <li><a href="#global-importance">2. Global Feature Importance</a></li>
        <li><a href="#individual-features">3. Individual Feature Analysis</a></li>
        <li><a href="#interactions">4. Feature Interactions</a></li>
        <li><a href="#local-explanations">5. Local Explanations</a></li>
        <li><a href="#hyperparameter-tuning">6. Hyperparameter Tuning</a></li>
    </ul>
</div>
"""

TOC_SECTION_WITH_FS = """
<div class="toc">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#performance">1. Prediction Performance</a></li>
        <li><a href="#global-importance">2. Global Feature Importance</a></li>
        <li><a href="#individual-features">3. Individual Feature Analysis</a></li>
        <li><a href="#interactions">4. Feature Interactions</a></li>
        <li><a href="#local-explanations">5. Local Explanations</a></li>
        <li><a href="#hyperparameter-tuning">6. Hyperparameter Tuning</a></li>
        <li><a href="#feature-selection">7. Feature Subset Selection (Backward)</a></li>
    </ul>
</div>
"""



PERFORMANCE_SECTION_START = """
<h2 id="performance">1. Prediction Performance</h2>
<p>{performance_description}</p>
"""

EVAL_NOTE_TRAIN_TEST = """"""

EVAL_NOTE_CV_ONLY = """
    <div class="alert alert-info">
        <strong>ℹ Cross-validation only:</strong> No holdout test set was used.
        Metrics are aggregated from out-of-fold predictions across all CV folds.
    </div>
"""

EVAL_NOTE_TRAIN_ONLY = """
    <div class="alert alert-warning">
        <strong>⚠ Training-set evaluation:</strong> The model was evaluated on the
        same data it was trained on. These metrics are optimistic and may overestimate
        performance on unseen data.
    </div>
"""

METRICS_TABLE_START = """
<h3>Performance Metrics</h3>
<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
"""

METRICS_TABLE_ROW = """
        <tr>
            <td>{metric_name}</td>
            <td class="metric-value">{metric_value}</td>
        </tr>
"""

METRICS_TABLE_END = """
    </tbody>
</table>
"""

PLOTS_SECTION_START = """
<h3>Validation Plots</h3>
<div class="plot-grid">
"""

PLOT_ITEM = """
    <div class="plot-grid-item">
        <img src="data:image/png;base64,{img_base64}" alt="{alt_text}">
    </div>
"""

PLOTS_SECTION_END = """
</div>
"""

GLOBAL_IMPORTANCE_SECTION = """
<h2 id="global-importance">2. Global Feature Importance</h2>
<p>Features ranked by their overall contribution to model predictions.</p>
<div class="plot-container">
    <img src="data:image/png;base64,{importance_plot}" alt="Feature Importance">
</div>
"""

INDIVIDUAL_FEATURES_SECTION_START = """
<h2 id="individual-features">3. Individual Feature Analysis</h2>
<p>Shape functions showing how each feature affects predictions. EBM learns these relationships directly from data.</p>
<div class="alert alert-info">
    <strong>Note:</strong> Each plot shows the learned relationship between the feature value and its contribution to the prediction.
</div>
"""

FEATURE_SHAPE_SECTION = """
<div class="feature-section">
    <h3>{feature_name}</h3>
    <p><strong>Importance Score:</strong> {importance_score}</p>
    <div class="plot-grid">
        <div class="plot-grid-item">
            <h4>Shape Function (Partial Dependence)</h4>
            <img src="data:image/png;base64,{shape_plot}" alt="Shape function for {feature_name}">
        </div>
        <div class="plot-grid-item">
            <h4>Feature Distribution / Density</h4>
            {density_plot_html}
        </div>
    </div>
</div>
"""

INTERACTIONS_SECTION_START = """
<h2 id="interactions">4. Feature Interactions</h2>
<p>Pairwise feature interactions detected by the EBM model. These show how features work together to influence predictions.</p>
"""

INTERACTION_ITEM = """
<div class="interaction-section">
    <h3>{interaction_name}</h3>
    <p><strong>Interaction Strength:</strong> {interaction_score}</p>
    <div class="plot-container">
        <img src="data:image/png;base64,{interaction_plot}" alt="Interaction: {interaction_name}">
    </div>
</div>
"""

LOCAL_SECTION_START = """
<h2 id="local-explanations">5. Local Explanations</h2>
<p>Individual predictions with feature-level contributions showing why the model made specific predictions.</p>
"""

LOCAL_EXPLANATION_ITEM = """
<div class="feature-section">
    <h3>Sample {sample_idx}</h3>
    <p><strong>True Value:</strong> {true_value} | <strong>Predicted:</strong> {pred_value}</p>
    <div class="plot-container">
        <img src="data:image/png;base64,{local_plot}" alt="Local explanation for sample {sample_idx}">
    </div>
</div>
"""

NO_DATA_MESSAGE = """
<div class="alert alert-warning">
    <strong>No data available for this section.</strong>
</div>
"""

POSITIVE_CLASS_LOCAL_SECTION_START = """
<h2 id="positive-class-explanations">{section_number}. Positive Class — Local Explanations</h2>
<p>Local explanations for every datapoint predicted in the positive class
(predicted label&nbsp;=&nbsp;1). Each bar chart shows the per-feature contribution
that pushed the model towards a positive prediction for that individual.</p>
<div class="alert alert-info">
    <strong>Note:</strong> {n_positive} sample(s) predicted as positive class.
</div>
"""

# ── Feature Selection Section Templates ──────────────────────────────

FEATURE_SELECTION_SECTION_START = """
<h2 id="feature-selection">{section_number}. Feature Subset Selection (Backward Elimination)</h2>
<p>Backward elimination was used to find a minimal feature subset that maintains
near-equivalent prediction performance. At each step the least important feature
(according to the model's <code>term_importances()</code>) is removed, the model
is retrained, and the CV score is compared to the all-features baseline.</p>
"""

FEATURE_SELECTION_SUMMARY = """
<div class="summary">
    <h3>Selection Summary</h3>
    <p><strong>Original features:</strong> {n_original}</p>
    <p><strong>Selected features:</strong> {n_selected}</p>
    <p><strong>Tolerance:</strong> {tolerance}</p>
    <p><strong>Baseline CV score:</strong> {baseline_score}</p>
    <p><strong>Final CV score:</strong> {final_score}</p>
    <p><strong>Relative drop:</strong> {relative_drop}</p>
</div>
"""

FEATURE_SELECTION_SELECTED_LIST = """
<h3>Selected Features</h3>
<div class="alert alert-info">
    {feature_badges}
</div>
"""

FEATURE_SELECTION_HISTORY_TABLE_START = """
<h3>Elimination History</h3>
<table>
    <thead>
        <tr>
            <th>Step</th>
            <th># Features</th>
            <th>Removed Feature</th>
            <th>CV Score</th>
            <th>Delta</th>
            <th>Relative Drop</th>
        </tr>
    </thead>
    <tbody>
"""

FEATURE_SELECTION_HISTORY_ROW = """
        <tr>
            <td>{step}</td>
            <td>{n_features}</td>
            <td>{removed_feature}</td>
            <td class="metric-value">{cv_score}</td>
            <td>{delta}</td>
            <td>{relative_drop}</td>
        </tr>
"""

FEATURE_SELECTION_HISTORY_TABLE_END = """
    </tbody>
</table>
"""

FEATURE_SELECTION_PLOT = """
<h3>Elimination Curve</h3>
<p>CV score as features are removed. The dashed line marks the tolerance band around the baseline.</p>
<div class="plot-container">
    <img src="data:image/png;base64,{elimination_plot}" alt="Feature Elimination Curve">
</div>
"""


# ── Hyperparameter Tuning Templates ──────────────────────────────────
TUNING_SECTION_START = """
<h2 id="hyperparameter-tuning">{section_number}. Hyperparameter Tuning Results</h2>
<p>The model was tuned using {method} with {n_candidates} candidates and {n_folds}-fold cross-validation. 
The best parameters found were:</p>
<div class="alert alert-info">
    {best_params_display}
</div>
<p>The table below shows all evaluated parameter combinations, ranked by their mean test score.</p>
"""

TUNING_TABLE_START = """
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Mean Score</th>
            <th>Std Score</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
"""

TUNING_TABLE_ROW = """
        <tr>
            <td>#{rank}</td>
            <td class="metric-value">{mean_score:.6f}</td>
            <td>&plusmn;{std_score:.6f}</td>
            <td><code>{params}</code></td>
        </tr>
"""

TUNING_TABLE_END = """
    </tbody>
</table>
"""
