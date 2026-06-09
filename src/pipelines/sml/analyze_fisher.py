"""
@author: Radosław Pławecki
"""

import pandas as pd
from scipy.stats import fisher_exact 
from statsmodels.stats.multitest import multipletests


class FisherAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def run(self, X, y, feature_names):
        p_values = []

        for i in range(X.shape[1]):
            table = pd.crosstab(X[:, i], y)

            if table.shape != (2, 2):
                p_values.append(1.0)
            else:
                _, p = fisher_exact(table)
                p_values.append(p)

        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=self.alpha, method='fdr_bh'
        )

        return [
            (feature_names[i], pvals_corrected[i])
            for i in range(len(feature_names)) if reject[i]
        ]
