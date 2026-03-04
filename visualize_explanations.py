"""
Visualize local explanation vectors from predictions.csv using t-SNE.

Reads a predictions.csv file produced by the EBM runner (with explain_* columns),
applies t-SNE dimensionality reduction, appends projected coordinates back to the CSV,
and generates an HTML report with colored scatter plots.

Usage examples
--------------
  # Default (no normalization)
  python visualize_explanations.py --predictions ./ebm_output/predictions.csv

  # With StandardScaler normalization
  python visualize_explanations.py --predictions ./ebm_output/predictions.csv --normalize standard

  # With MinMaxScaler normalization
  python visualize_explanations.py --predictions ./ebm_output/predictions.csv --normalize minmax

  # Custom output directory
  python visualize_explanations.py --predictions ./ebm_output/predictions.csv --output-dir ./viz_output
"""

import argparse
import io
import base64
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _is_continuous(labels: np.ndarray) -> bool:
    """Return True if labels should be treated as continuous (regression) values."""
    # If dtype is float and there are many unique values, treat as continuous
    try:
        float_labels = labels.astype(float)
        n_unique = len(np.unique(float_labels))
        # Heuristic: if float dtype and more than 20 unique values, treat as continuous
        if np.issubdtype(labels.dtype, np.floating) or (
            n_unique > 20 and np.issubdtype(labels.dtype, np.number)
        ):
            return True
    except (ValueError, TypeError):
        pass
    return False


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Create a scatter plot colored by label.

    For classification (discrete labels with <= 20 unique values), uses a
    qualitative colormap. For regression (continuous/float labels or > 20
    unique values), uses the 'viridis' continuous colormap.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)

    if _is_continuous(labels):
        # Continuous (regression) path: use viridis colormap
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels.astype(float),
            s=18,
            alpha=0.7,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label="Target value")
    elif len(unique_labels) <= 20:
        # Discrete classification path: qualitative colors + legend
        cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.7,
                label=str(lbl),
                color=cmap(i / max(len(unique_labels) - 1, 1)),
                edgecolors="white",
                linewidths=0.3,
            )
        ax.legend(title="Label", fontsize=8, markerscale=1.5, framealpha=0.9)
    else:
        # Many discrete categories: fall back to continuous colormap
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels.astype(float),
            s=18,
            alpha=0.7,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label="Label")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ── HTML template ────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Explanation Projections</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto; background: white;
            padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50; font-size: 2.2em; margin-bottom: 10px;
            border-bottom: 4px solid #3498db; padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e; font-size: 1.6em; margin-top: 40px;
            margin-bottom: 20px; padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        .summary {{
            background: #ecf0f1; padding: 20px; border-radius: 5px;
            margin-bottom: 30px;
        }}
        .summary p {{ margin: 8px 0; font-size: 1.05em; }}
        .summary strong {{ color: #2c3e50; }}
        .plot-container {{
            margin: 30px 0; text-align: center;
        }}
        .plot-container img {{
            max-width: 100%; height: auto; border: 1px solid #ddd;
            border-radius: 4px; padding: 5px; background: white;
        }}
        .alert {{
            padding: 15px; margin: 20px 0; border-radius: 4px;
        }}
        .alert-info {{
            background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;
        }}
        .alert-warning {{
            background: #fff3cd; border-left: 4px solid #856404; color: #856404;
        }}
        code {{
            background: #f4f4f4; padding: 2px 6px; border-radius: 3px;
            font-family: 'Courier New', monospace; font-size: 0.9em;
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize local explanation vectors with t-SNE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.csv (must contain explain_* columns).",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "standard", "minmax"],
        default="none",
        help="Normalization to apply to explanation vectors before projection (default: none).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the HTML report and updated CSV. Defaults to same directory as predictions.csv.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="y_true",
        help="Column to use for coloring points (default: y_true).",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    if not os.path.isfile(args.predictions):
        print(f"Error: file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.predictions)

    # Identify explanation columns
    explain_cols = [c for c in df.columns if c.startswith("explain_")]
    if not explain_cols:
        print(
            "Error: no explain_* columns found in predictions.csv.\n"
            "Run the EBM runner with --save-predictions first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.label_column not in df.columns:
        print(
            f"Error: label column '{args.label_column}' not found. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.predictions))
    os.makedirs(output_dir, exist_ok=True)

    X_explain = df[explain_cols].values.astype(float)
    labels = df[args.label_column].values

    # Drop rows with NaN explanations
    valid_mask = ~np.isnan(X_explain).any(axis=1)
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"Warning: dropping {n_dropped} row(s) with NaN explanation values.")
        X_explain = X_explain[valid_mask]
        labels = labels[valid_mask]
        df = df[valid_mask].copy()

    n_samples, n_features = X_explain.shape
    print(f"Loaded {n_samples} samples × {n_features} explanation features")

    # ── Normalize ────────────────────────────────────────────────────
    normalize_label = args.normalize
    if normalize_label == "standard":
        scaler = StandardScaler()
        X_explain = scaler.fit_transform(X_explain)
        print("Applied StandardScaler normalization")
    elif normalize_label == "minmax":
        scaler = MinMaxScaler()
        X_explain = scaler.fit_transform(X_explain)
        print("Applied MinMaxScaler normalization")
    else:
        print("No normalization applied")

    # ── t-SNE ────────────────────────────────────────────────────────
    perplexity = min(args.perplexity, n_samples - 1)
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.random_state,
        max_iter=1000,
    )
    tsne_coords = tsne.fit_transform(X_explain)
    df["tsne_1"] = np.nan
    df["tsne_2"] = np.nan
    df.loc[df.index, "tsne_1"] = tsne_coords[:, 0]
    df.loc[df.index, "tsne_2"] = tsne_coords[:, 1]
    print("  t-SNE done.")

    # ── Save updated CSV ─────────────────────────────────────────────
    out_csv = os.path.join(output_dir, "predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved updated predictions CSV: {out_csv}")

    # ── Generate t-SNE plot ──────────────────────────────────────────
    fig_tsne = _scatter_plot(
        tsne_coords, labels,
        title="t-SNE of Local Explanation Vectors",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
    )
    tsne_b64 = _fig_to_base64(fig_tsne)

    # ── Build HTML report ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_type = "regression" if _is_continuous(labels) else "classification"
    parts = []
    parts.append(f"""
<h1>Local Explanation Projections</h1>
<p class="timestamp">Generated: {timestamp}</p>
""")

    parts.append(f"""
<div class="summary">
    <h2>Summary</h2>
    <p><strong>Input file:</strong> <code>{os.path.basename(args.predictions)}</code></p>
    <p><strong>Samples:</strong> {n_samples}</p>
    <p><strong>Explanation features:</strong> {n_features}</p>
    <p><strong>Normalization:</strong> {normalize_label}</p>
    <p><strong>Label column:</strong> <code>{args.label_column}</code></p>
    <p><strong>Task type:</strong> {task_type}</p>
    <p><strong>Unique labels:</strong> {len(np.unique(labels))}</p>
</div>
""")

    parts.append("""
<h2>t-SNE Projection</h2>
<p>t-SNE projects the local explanation vectors into 2D, preserving
local neighbourhood structure. Points are colored by the label column.</p>
""")
    parts.append(f"""
<div class="plot-container">
    <img src="data:image/png;base64,{tsne_b64}" alt="t-SNE projection">
</div>
""")

    content = "\n".join(parts)
    html = HTML_TEMPLATE.format(content=content)

    html_path = os.path.join(output_dir, "explanation_projections.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML report: {html_path}")


if __name__ == "__main__":
    main()
