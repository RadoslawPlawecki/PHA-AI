"""
@author: Radosław Pławecki
"""

from ..plots_formatting import use_latex
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

use_latex()

df = pd.read_csv("metadata.csv", delimiter=';', usecols=['sample_id', 'checkv_vib_total', 'checkv_vib_good', 
                                                         'checkv_vs2_total', 'checkv_vs2_good',
                                                         'checkv_gen_total', 'checkv_gen_good'])

methods_map = {'vib': 'VIBRANT', 'vs2': 'VirSorter2', 'gen': 'geNomad'}
plot_data = []

for prefix, name in methods_map.items():
    temp = df[['sample_id', f'checkv_{prefix}_total', f'checkv_{prefix}_good']].copy()
    temp.columns = ['Sample', 'Total', 'Good']
    temp['Method'] = name
    plot_data.append(temp)

df_long = pd.concat(plot_data).reset_index(drop=True)

all_samples = df['sample_id'].unique()
mid_point = len(all_samples) // 2
groups = [all_samples[:mid_point], all_samples[mid_point:]]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

palette_main = {
    'VIBRANT': '#66C2A5',
    'VirSorter2': '#FC8D62',
    'geNomad': '#8DA0CB'
}

palette_light = {
    'VIBRANT': '#BFE7DB',
    'VirSorter2': '#FDD1C2',
    'geNomad': '#CCD5EE'
}

for i, sample_group in enumerate(groups):
    ax = axes[i]
    subset = df_long[df_long['Sample'].isin(sample_group)]

    sns.barplot(
        data=subset, x='Sample', y='Total', hue='Method',
        palette=palette_light,
        ax=ax, alpha=1.0
    )
    
    sns.barplot(
        data=subset, x='Sample', y='Good', hue='Method',
        palette=palette_main,
        ax=ax, alpha=1.0
    )
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.get_legend().remove()

fig.suptitle(r'Number of putative viral genomes', fontsize=16, y=0.96)

for ax in axes:
    ax.set_ylabel('Number of putative viral genomes', fontsize=14, labelpad=10)
    ax.set_xlabel('Sample', fontsize=14, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#66C2A5', lw=4, label='VIBRANT (Good/Total)'),
    Line2D([0], [0], color='#FC8D62', lw=4, label='VirSorter2 (Good/Total)'),
    Line2D([0], [0], color='#8DA0CB', lw=4, label='geNomad (Good/Total)')
]

axes[0].legend(handles=custom_lines, loc='upper right', fontsize=10, frameon=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('plots/metadata/total_good_checkv_bar.pdf') 
plt.show()
