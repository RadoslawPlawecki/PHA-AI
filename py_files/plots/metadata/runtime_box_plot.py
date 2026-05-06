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
sns.set_style('whitegrid')

ax = sns.boxplot(
    x='Tool', 
    y='Runtime [min]', 
    data=df_melted,
    hue='Tool',  
    palette='Set2',
    legend=False,   
    width=0.5,
    linewidth=1.2,
    flierprops={"marker": "x"}
)

plt.title('Genome assembly runtimes', fontsize=14)
plt.xlabel('Tool', fontsize=12)
plt.ylabel('Runtime [min]', fontsize=12)

plt.tight_layout()
plt.savefig('plots/metadata/runtime_box_plot.pdf') 
plt.show()
