"""
@author: Radosław Pławecki
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..plots.plots_formatting import use_latex
from scipy.stats import shapiro, ttest_ind

use_latex()

tool = 'VS2'
data_path = f'data/ml/{tool}_vC2.csv'

df = pd.read_csv(data_path, delimiter=',')

label = df["label"]
df = df.drop(columns='label')

df = df.copy()

df["richness"] = df.drop(columns="id").sum(axis=1)

df = df[["id", "richness"]]

df = pd.concat([df, label], axis=1)

df = df.sort_values(
    by="id",
    key=lambda col: col.str.extract(r'(\d+)').astype(int)[0]
).reset_index(drop=True)

# df.to_csv(f'experiments/diversity_analysis/{tool}_alp_div.csv', sep=';', index=False)

print(df.head())

print("\n--- STATISTICS ---")
print(df.groupby("label")["richness"].describe())

plt.figure()
sns.boxplot(x="label", y="richness", data=df)
plt.title("Richness vs label")
plt.show()

plt.figure()
sns.histplot(data=df, x="richness", hue="label", kde=True)
plt.show()

group0 = df[df["label"] == 0]["richness"]
group1 = df[df["label"] == 1]["richness"]

print("\n--- NORMALITY (Shapiro-Wilk) ---")
print("Label 0:", shapiro(group0))
print("Label 1:", shapiro(group1))

stat, p = ttest_ind(group0, group1)

print("\n--- T-STUDENT TEST ---")
print("Statistic:", stat)
print("p-value:", p)

if p < 0.05:
    print("\nResult: significant difference observed (p < 0.05).")
else:
    print("\nResult: no significant difference observed (p >= 0.05).")