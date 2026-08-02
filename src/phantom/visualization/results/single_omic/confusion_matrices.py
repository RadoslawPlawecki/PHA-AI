"""
@author: Radosław Pławecki
"""

from project.src.plots.plots_formatting import use_latex
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_confusion_matrix_grid(base_dir, validation_mode, model):
    use_latex()
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11), layout="constrained")
    rows = ['geN', 'VIB', 'VS2']
    cols = ['comp', 'host', 'func']
    rows_map = {'geN': 'geNomad', 'VIB': 'VIBRANT', 'VS2': 'VirSorter2'}
    cols_map = {'comp': 'Comp', 'host': 'Host', 'func': 'Func'}
    validation_map = {'loocv': 'Leave-One-Out CV', 'rcv': 'Repeated Stratified 5-Fold CV'}
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory {base_dir} not found!")
    all_folders = os.listdir(base_dir)
    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            ax = axes[r_idx, c_idx]
            target_folder = None
            for folder in all_folders:
                if folder.startswith(f"run_{model}_{r}_{c}_"):
                    target_folder = folder
                    break
            if target_folder:
                cm_path = os.path.join(base_dir, target_folder, validation_mode, "confusion_matrix.json")
                if os.path.exists(cm_path):
                    with open(cm_path, 'r') as f:
                        data = json.load(f)
                    try:
                        cm = np.array([
                            [data.get("TN", 0), data.get("FP", 0)],
                            [data.get("FN", 0), data.get("TP", 0)]
                        ])
                    except AttributeError:
                        print(f"Unexpected file format: {cm_path}")
                        ax.text(0.5, 0.5, "File format error", ha='center', va='center', color='red')
                        ax.axis('off')
                        continue
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                                annot_kws={'size': 20, 'weight': 'bold'}, ax=ax,
                                xticklabels=['0', '1'],
                                yticklabels=['0', '1'],
                                square=True, linewidths=1.5, linecolor='#979797')
                    ax.tick_params(axis='x', labelsize=16, labelrotation=0) 
                    ax.tick_params(axis='y', labelsize=16)
                    if c_idx == 0:
                        ax.set_ylabel(rows_map[r], fontsize=18, fontweight='bold', labelpad=15)
                    if c_idx == 2:
                        ax.set_ylabel("True Class", fontsize=17, fontweight='bold', rotation=270, labelpad=-188)
                    else:
                        ax.set_xlabel("")
                    if r_idx == 0:
                        ax.set_title(cols_map[c], fontsize=18, fontweight='bold', pad=15)
                    if r_idx == 2:
                        ax.set_xlabel("Predicted Class", fontsize=17, fontweight='bold', labelpad=15)
                    else:
                        ax.set_xlabel("")
                else:
                    ax.text(0.5, 0.5, f"File not found\nCM ({validation_mode})", ha='center', va='center', color='red')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"Directory not found\n{r}_{c}", ha='center', va='center', color='red')
                ax.axis('off')
    plt.suptitle(f"Confusion matrices: {validation_map[validation_mode]}", fontsize=21, fontweight='bold', y=0.96)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.35)
    output_filename = f"plots/results/single_omic/{model}/confusion_matrix_grid_{validation_mode}.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()

if __name__ == "__main__":
    model = "rf"
    results_path = f"data/results/single_omic/{model}"
    generate_confusion_matrix_grid(results_path, "loocv", model=model)
    generate_confusion_matrix_grid(results_path, "rcv", model=model)
    