"""
@author: Radosław Pławecki
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metric(df_long, metric_name="auc", method_name="loocv", model_name=None, savefig=None):
    from ..plots_formatting import use_latex
    use_latex()
    metrics_dict = {
        "auc": "Area Under the Curve",
        "balanced_accuracy": "Balanced Accurary",
        "f1": "F1-score",
        "precision": "Precision", 
        "recall": "Recall",
        "specificity": "Specificity",
        "gmean": "Geometric Mean",
        "nvp": "Negative Predictive Value",
        "pr_auc": "Precision-Recall Area Under the Curve",
        "mcc": "Matthews Correlation Coefficient"
    }
    gtool_dict = {
        "geN": "geNomad",
        "VIB": "VIBRANT",
        "VS2": "VirSorter2"
    }
    method_name_dict = {
        "loocv": "Leave-One-Out Cross-Validation",
        "rcv": "Repeated 5-Fold Cross-Validation"
    }
    modality_dict = {
        "comp": "Composition",
        "host": "Host",
        "func": "Function"
    }
    plot_df = df_long[
            (df_long["metric"] == metric_name)
            & (df_long["method"] == method_name)
    ].copy()

    plot_df["gtool"] = pd.Categorical(
        plot_df["gtool"],
        categories=["geN", "VIB", "VS2"],
        ordered=True
    )

    plot_df["modality"] = pd.Categorical(
        plot_df["modality"],
        categories=["host", "func", "comp"],
        ordered=True
    )

    heatmap_df = (
        plot_df
        .pivot(
            index="gtool",
            columns="modality",
            values="value"
        )
        .rename(index=gtool_dict)
        .rename(columns=modality_dict)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        linewidths=0.5,
        cbar_kws={"label": metrics_dict[metric_name]},
        annot_kws={
            "size": 16,     
        },
        ax=ax
    )

    plt.title(f"{method_name_dict[method_name]}: {metrics_dict[metric_name]}", fontsize=16, pad=10)

    ax.set_xlabel("Modality", fontsize=14, labelpad=15)
    ax.set_ylabel("Virus Identification Tool", fontsize=14, labelpad=15)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        ha="center",
        rotation=0,
        fontsize=12
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=12
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(metrics_dict[metric_name], fontsize=14, labelpad=20)

    plt.tight_layout()
    # fig.subplots_adjust(left=0.1)
    if savefig:
        plt.savefig(f"plots/sml/{model_name}/{method_name}_{metric_name}_{model_name}.pdf", format='pdf')
    plt.close()
    # plt.show()
