"""
Grouped bar-chart plots comparing healthy vs. allergic patients on
taxonomic/host diversity metrics, one subplot per virus-identification tool.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phantom.visualization.plots_formatting import use_latex

GROUP_COLORS = {0: '#a1d99b', 1: '#238b45'}
GROUP_LABELS = {0: 'Healthy', 1: 'Allergy'}
VTOOL_DISPLAY_NAMES = {"geN": "geNomad", "VIB": "VIBRANT", "VS2": "VirSorter2"}


class DiversityVisualizer:
    @staticmethod
    def plot_group_bar_by_vtool(
        group_stats_by_vtool: dict[str, pd.DataFrame], value_col: str,
        title: str, ylabel: str, save_path: Path | str | None = None
    ) -> None:
        """
        group_stats_by_vtool[vtool] is indexed by label (0/1) with
        MultiIndex columns (metric, "mean"/"std") -- e.g. produced by
        `per_sample.groupby("label")[[...]].agg(["mean", "std"])`. Bars
        show the group mean with a standard-deviation whisker on top.
        """
        use_latex()
        vtools = list(group_stats_by_vtool)
        fig, axes = plt.subplots(1, len(vtools), figsize=(5 * len(vtools), 5), sharey=True)
        if len(vtools) == 1:
            axes = [axes]
        for ax, vtool in zip(axes, vtools):
            stats = group_stats_by_vtool[vtool]
            labels = [0, 1]
            means = [
                stats.loc[label, (value_col, "mean")] if label in stats.index else 0
                for label in labels
            ]
            stds = np.nan_to_num(
                [
                    stats.loc[label, (value_col, "std")] if label in stats.index else 0
                    for label in labels
                ],
                nan=0.0,
            )
            ax.bar(
                [GROUP_LABELS[label] for label in labels], means,
                yerr=stds, capsize=6, ecolor='black',
                color=[GROUP_COLORS[label] for label in labels],
                edgecolor='black', linewidth=0.8,
            )
            ax.set_title(VTOOL_DISPLAY_NAMES.get(vtool, vtool), fontsize=13, weight='bold', pad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(ylabel, fontsize=12, labelpad=8)
        plt.suptitle(title, fontsize=15, weight='bold', y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close()
