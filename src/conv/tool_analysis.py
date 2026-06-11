import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    shapiro
)

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from ..common.logger import log_tqdm


class ToolAnalysis:
    def __init__(self, processor, logger):
        self.processor = processor
        self.logger = logger

    def _interpret_pvalue(self, pvalue, alpha=0.05):
        if pvalue < alpha:
            return (
                "Statistically significant "
                "difference detected."
            )
        return (
            "No statistically significant "
            "difference detected."
        )

    def _interpret_kendall(self, w):
        if w < 0.1:
            return "Negligible effect size"
        if w < 0.3:
            return "Small effect size"
        if w < 0.5:
            return "Moderate effect size"
        return "Large effect size"

    def _kendalls_w(self, statistic, n_subjects, n_conditions):
        return statistic / (
            n_subjects * (n_conditions - 1)
        )

    def _log_result(self, title, statistic, pvalue, effect_size=None, effect_label=None):
        interpretation = self._interpret_pvalue(pvalue)
        message = (
            f"\n[{title}]\n"
            f"Statistic: {statistic:.6f}\n"
            f"P-value: {pvalue:.8f}\n"
            f"Interpretation: "
            f"{interpretation}\n"
        )
        if effect_size is not None:
            effect_interpretation = (
                self._interpret_kendall(effect_size)
            )
            message += (
                f"{effect_label}: "
                f"{effect_size:.6f}\n"
                f"Effect interpretation: "
                f"{effect_interpretation}\n"
            )
        log_tqdm(
            self.logger,
            message
        )

    def descriptive_statistics(self):
        log_tqdm(self.logger, "[TOOLS] Descriptive statistics")
        df = (self.processor.get_tool_dataframe())
        result = df.describe().T
        result["median"] = df.median()
        log_tqdm(self.logger, f"\n{result}\n")
        return result

    def normality_tests(self):
        log_tqdm(self.logger, "[TOOLS] Shapiro-Wilk tests")
        results = []
        for tool in self.processor.config.tools:
            stat, p = shapiro(self.processor.df[tool])
            interpretation = (
                "Data likely normally distributed."
                if p > 0.05
                else
                "Data likely NOT normally distributed."
            )
            log_tqdm(
                self.logger,
                (
                    f"\n[{tool}]\n"
                    f"Shapiro statistic: "
                    f"{stat:.6f}\n"
                    f"P-value: {p:.8f}\n"
                    f"Interpretation: "
                    f"{interpretation}\n"
                )
            )
            results.append({
                "tool": tool,
                "statistic": stat,
                "pvalue": p
            })
        return pd.DataFrame(results)

    def friedman_test(self):
        log_tqdm(self.logger, "[TOOLS] Friedman test")
        df = self.processor.df
        stat, p = friedmanchisquare(
            df["genomad"],
            df["virsorter2"],
            df["vibrant"]
        )
        n_subjects = len(df)
        n_conditions = len(
            self.processor.config.tools
        )
        kendalls_w = self._kendalls_w(
            stat,
            n_subjects,
            n_conditions
        )
        self._log_result(
            title="Friedman Test",
            statistic=stat,
            pvalue=p,
            effect_size=kendalls_w,
            effect_label="Kendall's W"
        )
        return pd.DataFrame([{
            "statistic": stat,
            "pvalue": p,
            "kendalls_w": kendalls_w
        }])

    def repeated_measures_anova(self):
        log_tqdm(self.logger, "[TOOLS] RM-ANOVA")
        long_df = (
            self.processor.get_long_dataframe()
        )
        model = AnovaRM(
            long_df,
            depvar="value",
            subject="sample",
            within=["tool"]
        )
        result = model.fit()
        log_tqdm(
            self.logger,
            f"\n{result}\n"
        )
        return result

    def posthoc_tests(self):
        log_tqdm(self.logger, "[TOOLS] Posthoc Wilcoxon tests")
        df = self.processor.df
        comparisons = [
            ("genomad", "virsorter2"),
            ("genomad", "vibrant"),
            ("virsorter2", "vibrant")
        ]
        results = []
        pvalues = []
        for a, b in comparisons:
            stat, p = wilcoxon(
                df[a],
                df[b]
            )
            pvalues.append(p)
            results.append({
                "comparison": f"{a} vs {b}",
                "statistic": stat,
                "raw_pvalue": p
            })
        corrected = multipletests(
            pvalues,
            method="fdr_bh"
        )[1]
        for i in range(len(results)):
            results[i][
                "adjusted_pvalue"
            ] = corrected[i]
            interpretation = (
                self._interpret_pvalue(
                    corrected[i]
                )
            )
            log_tqdm(
                self.logger,
                (
                    f"\n[POSTHOC] "
                    f"{results[i]['comparison']}\n"
                    f"Statistic: "
                    f"{results[i]['statistic']:.6f}\n"
                    f"Adjusted p-value: "
                    f"{corrected[i]:.8f}\n"
                    f"Interpretation: "
                    f"{interpretation}\n"
                )
            )
        return pd.DataFrame(results)
