"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np


class NearZeroVarianceFilter:
    def __init__(self, threshold: float = 1e-6, logger=None):
        self.threshold = threshold
        self.logger = logger

    def fit_transform(self, X: pd.DataFrame):
        values, feature_names = X.values, X.columns
        mask = values.var(axis=0) > self.threshold
        n_before = values.shape[1]
        n_after = mask.sum()
        n_removed = n_before - n_after
        self._log_filtering(X, mask, n_before, n_after, n_removed)
        return values[:, mask], feature_names[mask]

    def _log_filtering(self, X, mask, n_before: int, n_after: int, n_removed: int) -> None:
        msg1 = f"Near-zero variance filter (threshold={self.threshold:g})"
        msg2 = (
            f"Features: {n_before} -> {n_after} "
            f"(removed {n_removed}, {100 * n_removed / n_before:.1f}%)"
        )
        removed_features = X.columns[~mask]
        if len(removed_features) > 0:
            msg3 = f"Removed features: {', '.join(removed_features)}"
        else:
            msg3 = "No features removed"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
            self.logger.info(msg3)
        else:
            print(msg1)
            print(msg2)
            print(msg3)
