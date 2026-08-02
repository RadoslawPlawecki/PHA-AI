"""
@author: Radosław Pławecki
"""

from project.src.plots.plots_formatting import use_latex
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric(metric_name, datasets_dict, output_path="metric_comparison.pdf"):
    use_latex()
    plt.figure(figsize=(16, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12']
    metric_lower = metric_name.lower()
    data_plotted = False
    for idx, (label, filepath) in enumerate(datasets_dict.items()):
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File does not exist ({filepath}). Skipping.")
            continue
        try:
            df = pd.read_csv(filepath, sep=';')
            df['Metric'] = df['Metric'].astype(str).str.lower().str.strip()
            if metric_lower not in df['Metric'].values:
                print(f"⚠️ Warning: Metric '{metric_name}' not found in {filepath}.")
                continue
            row = df[df['Metric'] == metric_lower].iloc[0]
            columns_to_plot = [col for col in df.columns if col not in ['Metric']]
            x_labels = columns_to_plot
            y_values = row[columns_to_plot].astype(float)
            plt.plot(x_labels, y_values, 
                        marker=markers[idx % len(markers)], 
                        color=colors[idx % len(colors)],
                        linewidth=2.5, markersize=9, label=label)
            data_plotted = True
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
    if not data_plotted:
        print("❌ Failed to generate plot. Check file paths and metric names.")
        plt.close()
        return
    plt.title(f"({metric_name.upper()}): Performance Comparison", 
              fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.ylabel(f"{metric_name.upper()}", fontsize=13, fontweight='bold', labelpad=10)
    plt.xlabel("Configuration", fontsize=13, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="Tool", title_fontsize='12', fontsize='11', 
               loc='lower left', bbox_to_anchor=(0, 0), frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Plot saved successfully as: {output_path}")


if __name__ == "__main__":
    multiple_datasets = {
        "iPHoP": "data/results/multi_omic/host=iphop/catboost/all_metrics.csv",
        "CHERRY": "data/results/multi_omic/host=cherry/catboost/all_metrics.csv",
    }
    plot_metric("mcc", multiple_datasets, "plots/host_comparison.pdf")
