"""
@author: Radosław Pławecki
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metric(df_long, metric_name="auc", method_name="loocv"):
    from ..plots_formatting import use_latex

    metrics_dict = {
        "auc": "Area Under the Curve",
        "balanced_accuracy": "Balanced Accurary",
        "f1": "F-score",
        "precision": "precision", 
        "recall": "Recall",
        "specificity": "Specificity",
        "gmean": "Geometric Mean",
        "nvp": "Negative Predictive Value",
        "pr_auc": "Precision-Recall Area Under the Curve",
        "mcc": "Matthews Correlation Coefficient"
    }

    method_name_dict = {
        "loocv": "Leave-One-Out Cross-Validation",
        "rcv": "Repeated 5-Fold Cross-Validation"
    }

    use_latex()

    plot_df = df_long[
        (df_long["metric"] == metric_name) &
        (df_long["method"] == method_name)
    ]

    heatmap_df = plot_df.pivot(
        index="mP",
        columns="level",
        values="value"
    )

    n_cols = heatmap_df.shape[1]
    n_rows = heatmap_df.shape[0]

    fig_width = max(8, n_cols * 2.8)
    fig_height = max(4, n_rows * 1.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="Greens",
        annot_kws={"size": 14},
        cbar_kws={"label": metric_name},
        fmt=".3f"
    )

    plt.title(f"{method_name_dict[method_name]}: {metrics_dict[metric_name]}", fontsize=16, pad=10)

    ax.set_xlabel("Taxonomic level", fontsize=14, labelpad=10)
    ax.set_ylabel("Minimum number of shared patients", fontsize=14, labelpad=15)

    rotation = 0
    if n_cols > 5:
        rotation = 45

    ax.set_xticklabels(
        ax.get_xticklabels(),
        ha="center",
        rotation=rotation,
        fontsize=14
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=14
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(metrics_dict[metric_name], fontsize=14, labelpad=20)

    plt.tight_layout()
    fig.subplots_adjust(left=0.1)
    plt.show()