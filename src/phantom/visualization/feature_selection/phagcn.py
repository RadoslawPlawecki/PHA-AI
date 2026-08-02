"""
@author: Radosław Pławecki
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ..plots_formatting import use_latex


def phagcn_feat_select_plot(path_bin, path_nbin, label1="Model A", label2="Model B", out_dir="plots/feature_selection/phagcn"):
    use_latex()
    df1 = pd.read_csv(path_bin)
    df2 = pd.read_csv(path_nbin)
    tools = ['geN', 'VIB', 'VS2']
    tools_dict = {"geN": "geNomad", "VIB": "VIBRANT", "VS2": "VirSorter2"}
    methods = ['loocv', 'rcv']
    methods_dict = {"loocv": "LOOCV", "rcv": "Repeated 5-Fold CV"}
    metrics_dict = {
        "auc": "Area Under the Curve (AUC)", 
        "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1-score", 
        "precision": "Precision", 
        "recall": "Recall",
        "specificity": "Specificity", 
        "gmean": "Geometric Mean",
        "nvp": "Negative Predictive Value", 
        "pr_auc": "Precision-Recall AUC",
        "mcc": "Matthews Correlation Coefficient"
    }
    mp_values = [1, 2, 3, 4, 5]
    os.makedirs(out_dir, exist_ok=True)
    for metric in df1['metric'].unique():
        row1 = df1[df1['metric'] == metric]
        row2 = df2[df2['metric'] == metric]
        if row1.empty or row2.empty:
            continue
        metric_title = metrics_dict.get(metric, metric.replace('_', ' ').title())
        for method in methods:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            for idx, tool in enumerate(tools):
                ax = axes[idx]
                y1 = [row1[f"{method}_{tool}_MP{i}"].values[0] for i in mp_values]
                y2 = [row2[f"{method}_{tool}_MP{i}"].values[0] for i in mp_values]
                ax.plot(mp_values, y1, marker='o', markersize=6, linewidth=2, 
                        label=label1, color='#1b9e77') 
                ax.plot(mp_values, y2, marker='s', markersize=6, linewidth=2, 
                        linestyle='--', label=label2, color='#d95f02') 
                ax.set_title(tools_dict[tool], fontsize=14, weight='bold', pad=10)
                ax.set_xlabel("No. of shared samples", fontsize=12, labelpad=8)
                ax.set_xticks(mp_values)
                ax.set_xticklabels(mp_values, fontsize=11)
                ax.tick_params(axis='y', labelsize=11)
                ax.grid(True, linestyle=':', alpha=0.6) 
                if idx == 0:
                    ax.set_ylabel(metric_title, fontsize=13, weight='bold', labelpad=10)
                    ax.legend(fontsize=11, loc='best', frameon=True, title="Representation:")
            plt.suptitle(f"{methods_dict[method]}: {metric_title}", 
                         fontsize=16, weight='bold', y=0.98)
            plt.tight_layout()
            filename = f"{out_dir}/{metric}_{method}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            # plt.show()
            plt.close()


path_bin = "data/ml/sml/PGN_BIN_FEAT_SELECTION_CB.csv"
path_nbin = "data/ml/sml/PGN_NBIN_FEAT_SELECTION_CB.csv"
phagcn_feat_select_plot(path_bin=path_bin, path_nbin=path_nbin, label1="Binary", label2="Non-binary")
