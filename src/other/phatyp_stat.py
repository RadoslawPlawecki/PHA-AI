"""
@author: Radosław Pławecki
"""

import pandas as pd
from scipy.stats import mannwhitneyu


def compare_type_ratios(df: pd.DataFrame):
    df = df.copy()
    df["sample_num"] = (
        df["id"]
        .str.extract(r"\|S(\d+)", expand=False)
        .astype(int)
    )
    target = df[df["sample_num"] <= 34]
    control = df[df["sample_num"] > 34]
    results = []
    for col in ["temperate_ratio", "virulent_ratio"]:
        stat, pvalue = mannwhitneyu(
            control[col],
            target[col],
            alternative="two-sided"
        )
        results.append({
            "feature": col,
            "control_n": len(control),
            "target_n": len(target),
            "control_mean": round(control[col].mean(), 4),
            "target_mean": round(target[col].mean(), 4),
            "control_median": round(control[col].median(), 4),
            "target_median": round(target[col].median(), 4),
            "U_statistic": stat,
            "p_value": pvalue
        })
    return pd.DataFrame(results)


file_path = "data/modalities/2.0/features/phatyp/geN_PHT_FEAT.csv"
df = pd.read_csv(file_path, sep=';')
print(compare_type_ratios(df))
