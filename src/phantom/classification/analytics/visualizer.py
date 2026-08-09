"""
@author: Radosław Pławecki
"""

import matplotlib.pyplot as plt
from phantom.visualization.plots_formatting import use_latex


class Visualizer:
    @staticmethod
    def plot_unsupervised_grid(coords_dict, y_aligned, sample_ids, title_main="Multi-Dimensional Scaling", save_path=None):
        use_latex()
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        colors = ['#1f77b4' if label == 0 else '#d62728' for label in y_aligned]
        class_labels = ['Healthy (0)', 'Allergy (1)']
        config = {
            (0, 0): ("comp", "Phage genus (Jaccard)"),
            (0, 1): ("host", "Host-association scores (Euclidean)"),
            (1, 0): ("func", "Function ratios (Bray-Curtis)"),
            (1, 1): ("fused", "Modalities integrated")
        }
        for (row, col), (key, title) in config.items():
            ax = axs[row, col]
            coords = coords_dict[key]
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=200, alpha=0.85, edgecolors='k')
            for i, (x, y) in enumerate(coords):
                text_to_display = str(sample_ids[i]).split("|S")[1].strip() if sample_ids is not None else str(i + 1)
                ax.text(
                    x, y, text_to_display, 
                    fontsize=8, 
                    fontweight='bold', 
                    color='white', 
                    ha='center', 
                    va='center'
                )
            ax.set_title(title, fontsize=14, weight='bold', pad=10)
            ax.set_xlabel('MDS1', fontsize=12, labelpad=8)
            ax.set_ylabel('MDS2', fontsize=12, labelpad=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            for index, name in enumerate(class_labels):
                ax.scatter([], [], c=['#1f77b4', '#d62728'][index], label=name, edgecolors='k', s=75)
            ax.legend(loc='best')
        plt.suptitle(title_main, fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        # plt.show()
        plt.close()
