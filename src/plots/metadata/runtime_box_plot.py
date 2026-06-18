"""
@author: Radosław Pławecki
"""

from ..plots_formatting import use_latex
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

use_latex()

df = pd.read_csv("metadata.csv", delimiter=';', usecols=['vib_runtime', 'vs2_runtime', 'gen_runtime'])

df_melted = df.melt(var_name='Tool', value_name='Runtime [min]')

labels = {
    'vib_runtime': 'VIBRANT',
    'vs2_runtime': 'VirSorter2',
    'gen_runtime': 'geNomad'
}

df_melted['Tool'] = df_melted['Tool'].map(labels)

plt.figure(figsize=(8, 5))

palette = {
    'VIBRANT': '#66C2A5',
    'VirSorter2': '#FC8D62',
    'geNomad': '#8DA0CB'
}

ax = sns.boxplot(
    x='Tool', 
    y='Runtime [min]', 
    data=df_melted,
    hue='Tool',  
    palette=palette,
    legend=False,   
    width=0.5,
    linewidth=1.2,
    flierprops={"marker": "x"}
)

plt.title(r'\textbf{Virus identification tools runtimes}', fontsize=16, pad=10)
plt.xlabel('Tool', fontsize=14, labelpad=10)
plt.ylabel('Runtime [min]', fontsize=14, labelpad=10)

plt.tight_layout()
plt.savefig('plots/metadata/runtime_box_plot.pdf') 
plt.show()

print(sns.color_palette("Set2").as_hex())