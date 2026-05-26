"""
@author: Radosław Pławecki
"""

import os

import pandas as pd
import numpy as np

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    shapiro,
    mannwhitneyu,
    ttest_ind
)

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

from ..common.logger import setup_logger, log_tqdm

TOOLS = ["genomad", "virsorter2", "vibrant"]

def descriptive_stats(df, logger):
    log_tqdm(logger, "Computing descriptive statistics...")
    desc = df[TOOLS].describe().T
    desc["median"] = df[TOOLS].median()
    log_tqdm(logger, f"\n{desc}")
    return desc

def normality_tests(df, logger):

    log_tqdm(logger, "Running Shapiro-Wilk tests...")

    results = []

    for tool in TOOLS:

        stat, p = shapiro(df[tool])

        results.append({
            "tool": tool,
            "statistic": stat,
            "pvalue": p
        })

        log_tqdm(
            logger,
            f"{tool}: statistic={stat:.4f}, p={p:.6f}"
        )

    return pd.DataFrame(results)

def friedman_test(df, logger):

    log_tqdm(logger, "Running Friedman test...")

    stat, p = friedmanchisquare(
        df["genomad"],
        df["virsorter2"],
        df["vibrant"]
    )

    log_tqdm(
        logger,
        f"Friedman statistic={stat:.4f}, p={p:.8f}"
    )

    return stat, p

def repeated_measures_anova(df, logger):

    log_tqdm(logger, "Running repeated measures ANOVA...")

    long_df = df.melt(
        id_vars=["sample", "label"],
        value_vars=TOOLS,
        var_name="tool",
        value_name="value"
    )

    model = AnovaRM(
        long_df,
        depvar="value",
        subject="sample",
        within=["tool"]
    )

    result = model.fit()

    log_tqdm(logger, f"\n{result}")

    return result

def posthoc_tests(df, logger):

    log_tqdm(logger, "Running pairwise Wilcoxon tests...")

    comparisons = [
        ("genomad", "virsorter2"),
        ("genomad", "vibrant"),
        ("virsorter2", "vibrant")
    ]

    pvals = []
    results = []

    for a, b in comparisons:

        stat, p = wilcoxon(df[a], df[b])

        pvals.append(p)

        results.append({
            "comparison": f"{a} vs {b}",
            "statistic": stat,
            "raw_pvalue": p
        })

    corrected = multipletests(
        pvals,
        method="fdr_bh"
    )[1]

    for i in range(len(results)):
        results[i]["adjusted_pvalue"] = corrected[i]

    res_df = pd.DataFrame(results)

    log_tqdm(logger, f"\n{res_df}")

    return res_df

def analyze_labels(df, logger):

    log_tqdm(logger, "Running label analysis...")

    unique_labels = df["label"].unique()

    if len(unique_labels) < 2:

        log_tqdm(
            logger,
            "Only one label detected. Skipping analysis."
        )

        return None

    results = []

    for tool in TOOLS:

        group0 = df[df["label"] == 0][tool]
        group1 = df[df["label"] == 1][tool]

        p_norm0 = shapiro(group0)[1]
        p_norm1 = shapiro(group1)[1]

        normal = (
            p_norm0 > 0.05 and
            p_norm1 > 0.05
        )

        if normal:

            stat, p = ttest_ind(group0, group1)
            test_name = "t-test"

        else:

            stat, p = mannwhitneyu(group0, group1)
            test_name = "Mann-Whitney U"

        results.append({
            "tool": tool,
            "test": test_name,
            "statistic": stat,
            "pvalue": p
        })

        log_tqdm(
            logger,
            f"{tool}: {test_name}, p={p:.6f}"
        )

    return pd.DataFrame(results)