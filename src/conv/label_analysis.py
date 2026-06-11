import pandas as pd

from scipy.stats import (
    shapiro,
    mannwhitneyu,
    ttest_ind
)

from ..common.logger import log_tqdm


class LabelAnalysis:
    def __init__(self, processor, logger):
        self.processor = processor
        self.logger = logger

    def _select_test(self, group0, group1):
        p0 = shapiro(group0)[1]
        p1 = shapiro(group1)[1]
        normal = (
            p0 > 0.05 and
            p1 > 0.05
        )
        if normal:
            stat, p = ttest_ind(
                group0,
                group1
            )
            return "t-test", stat, p
        stat, p = mannwhitneyu(
            group0,
            group1
        )
        return "Mann-Whitney U", stat, p

    def run(self):
        log_tqdm(self.logger, "[LABEL] Label analysis")
        df = self.processor.df
        if len(df["label"].unique()) < 2:
            log_tqdm(
                self.logger,
                "Only one label detected."
            )
            return None
        results = []
        for tool in self.processor.config.tools:
            group0 = (
                df[df["label"] == 0][tool]
            )
            group1 = (
                df[df["label"] == 1][tool]
            )
            test_name, stat, p = (
                self._select_test(
                    group0,
                    group1
                )
            )
            results.append({
                "tool": tool,
                "test": test_name,
                "statistic": stat,
                "pvalue": p
            })
            interpretation = (
                "Label significantly affects results."
                if p < 0.05
                else
                "No significant effect of label detected."
            )

            log_tqdm(
                self.logger,
                (
                    f"\n[{tool}]\n"
                    f"Test: {test_name}\n"
                    f"Statistic: {stat:.6f}\n"
                    f"P-value: {p:.8f}\n"
                    f"Interpretation: "
                    f"{interpretation}\n"
                )
            )
        return pd.DataFrame(results)
