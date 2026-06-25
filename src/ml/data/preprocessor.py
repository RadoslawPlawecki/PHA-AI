"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np


class NearZeroVarianceFilter:
    def __init__(self, threshold: float = 1e-6, logger=None):
        self.threshold = threshold
        self.logger = logger

    def fit(self, X: pd.DataFrame):
        values = X.values
        self.mask_ = values.var(axis=0) > self.threshold
        self.feature_names_ = X.columns[self.mask_]
        n_before = values.shape[1]
        n_after = self.mask_.sum()
        n_removed = n_before - n_after
        self._log_filtering(X, self.mask_, n_before, n_after, n_removed)
        return self

    def transform(self, X: pd.DataFrame):
        if self.mask_ is None:
            raise ValueError("Filter must be fitted before calling transform().")
        return X.values[:, self.mask_], self.feature_names_

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def _log_filtering(self, X, mask, n_before: int, n_after: int, n_removed: int) -> None:
        msg1 = f"Near-Zero Variance Filter (Threshold={self.threshold:g})"
        msg2 = (
            f"Features: {n_before} -> {n_after} "
            f"(Removed {n_removed}, {100 * n_removed / n_before:.1f}%)"
        )
        removed_features = X.columns[~mask]
        if len(removed_features) > 0:
            msg3 = f"Removed Features: {', '.join(removed_features)}"
        else:
            msg3 = "No Features Removed"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
            self.logger.info(msg3)
        else:
            print(msg1)
            print(msg2)
            print(msg3)
