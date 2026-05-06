"""
@author: Radosław Pławecki
"""

from ..plots_formatting import use_latex
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

use_latex()

df = pd.read_csv("metadata.csv", delimiter=';', usecols=['sample_id', 'megahit_size_bytes', 
                                                         'vib_runtime', 'vs2_runtime', 
                                                         'gen_runtime'])

df_sorted = df.sort_values('megahit_size_bytes').reset_index(drop=True)

methods_map = {'vib_runtime': 'VIBRANT', 'vs2_runtime': 'VirSorter2', 'gen_runtime': 'geNomad'}
plot_data = []

for col, name in methods_map.items():
    temp = df_sorted[['sample_id', 'megahit_size_bytes', col]].copy()
    temp.columns = ['Sample', 'Size', 'Runtime']
    temp['Method'] = name
    plot_data.append(temp)

df_plot = pd.concat(plot_data).reset_index(drop=True)

plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

ax = sns.scatterplot(
    data=df_plot,
    x='Sample',
    y='Runtime',
    hue='Method',
    size='Size',
    sizes=(20, 500), # Zakres wielkości bąbelków
    alpha=0.6,
    palette='viridis',
    edgecolor='w',
    linewidth=0.5
)

plt.suptitle(r'Relation between file size and runtime', fontsize=16, y=0.96)
plt.xlabel(r'Samples (sorted ascending by file size)', fontsize=14, labelpad=10)
plt.ylabel(r'Runtime [min]', fontsize=14, labelpad=10)

h, l = ax.get_legend_handles_labels()
plt.legend(h[:4], l[:4], loc='upper left')

plt.xticks(rotation=90, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
# plt.savefig('plots/metodologia/runtime_bubble_plot.pdf')
plt.show()