"""
@author: Radosław Pławecki
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from ..plots_formatting import use_latex


def iphop_feat_select_plot(data_path, out_dir="plots/feature_selection/iphop"):
    use_latex()
    df = pd.read_csv(data_path)
    strategies = ["TSS_Z", "CLR_Z", "TSS", "RAW"]
    tools = ["geN", "VIB", "VS2"]
    tools_dict = {"geN": "geNomad", "VIB": "VIBRANT", "VS2": "VirSorter2"}
    methods_dict = {"loocv": "LOOCV", "rcv": "Repeated 5-Fold CV"}
    metrics_dict = {
        "auc": "Area Under the Curve (AUC)", "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1-score", "precision": "Precision", "recall": "Recall",
        "specificity": "Specificity", "gmean": "Geometric Mean",
        "nvp": "Negative Predictive Value", "pr_auc": "Precision-Recall AUC",
        "mcc": "Matthews Correlation Coefficient"
    }
    os.makedirs(out_dir, exist_ok=True)
    for _, row in df.iterrows():
        metric = row["metric"]
        metric_title = metrics_dict.get(metric, metric.replace('_', ' ').title())
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for idx, tool in enumerate(tools):
            ax = axes[idx]
            y_loocv = [row[f"loocv_{tool}_{strat}"] for strat in strategies]
            y_rcv = [row[f"rcv_{tool}_{strat}"] for strat in strategies]
            ax.plot(strategies, y_loocv, marker='o', markersize=6, linewidth=2, 
                        label="LOOCV", color='#1b9e77') 
            ax.plot(strategies, y_rcv, marker='s', markersize=6, linewidth=2, 
                    linestyle='--', label="RCV", color='#d95f02') 
            ax.set_title(tools_dict[tool], fontsize=14, weight='bold', pad=10)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlabel("Transformation Strategy", fontsize=12, labelpad=8)
            ax.set_xticklabels(strategies, fontsize=11)
            ax.set_xlim(left=-0.5, right=3.5)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            if idx == 0:
                ax.set_ylabel(metric_title, fontsize=13, weight='bold', labelpad=10)
                ax.legend(fontsize=11, loc='best', frameon=True)
        plt.suptitle(f"{metric_title}", 
                     fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        filename = f"{out_dir}/{metric}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.show()
        plt.close()


data_path = "data/ml/sml/IPH_FEAT_SELECTION_CB.csv"
iphop_feat_select_plot(data_path=data_path)
