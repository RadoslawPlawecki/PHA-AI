"""
@author: Radosław Pławecki
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_metrics(df_long, model_name, method_name, savefig=True):
    from ..plots_formatting import use_latex
    use_latex()
    metrics = ["auc", "balanced_accuracy", "f1", "precision", "recall", "specificity", "gmean", "nvp", "pr_auc", "mcc"]
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
        "loocv": "LOOCV",
        "rcv": "Repeated 5-Fold CV"
    }
    model_dict = {
        "rf": "Random Forest", 
        "catboost": "CatBoost"
    }
    modality_dict = {
        "comp": "Composition",
        "host": "Host",
        "func": "Function"
    }
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten() 
    for i, metric in enumerate(metrics):
        ax = axes[i]
        row_idx = i // 5 
        col_idx = i % 5   
        plot_df = df_long[
            (df_long["metric"] == metric)
            & (df_long["method"] == method_name)
        ].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No data for\n{metric}", ha='center', va='center', fontsize=11)
            ax.axis('off')
            continue
        plot_df["gtool"] = pd.Categorical(plot_df["gtool"], categories=["geN", "VIB", "VS2"], ordered=True)
        plot_df["modality"] = pd.Categorical(plot_df["modality"], categories=["host", "func", "comp"], ordered=True)
        heatmap_df = (
            plot_df
            .pivot(index="gtool", columns="modality", values="value")
            .rename(index=gtool_dict)
            .rename(columns=modality_dict)
        )
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="Greens",
            linewidths=0.5,
            annot_kws={"size": 12},
            square=True, 
            cbar_kws={"shrink": 0.75, "pad": 0.03},
            ax=ax
        )
        ax.set_title(metrics_dict[metric], fontsize=13, pad=12)
        ax.set_xlabel("Modality" if row_idx == 1 else "", fontsize=12, labelpad=8)
        if col_idx == 0:
            ax.set_ylabel("Viral Identification Tool", fontsize=12, labelpad=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([]) #
        ax.set_xticklabels(ax.get_xticklabels(), ha="center", rotation=0, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.15)
    plt.suptitle(
        f"{model_dict[model_name]} ({method_name_dict[method_name]})", 
        fontsize=22, y=0.94, weight='bold'
    )
    if savefig:
        out_dir = f"plots/sml/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/{method_name}_metrics_grid.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    