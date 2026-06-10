"""
@author: Radosław Pławecki
"""

import pandas as pd
from pipelines.common.data_loader import DataLoader 
from pipelines.common.preprocessor import NearZeroVarianceFilter 


def test_near_zero_variance_filter_features_removed():
    loader = DataLoader("project/tests/data/features/001_input.csv")
    X, _ = loader.load()
    filt = NearZeroVarianceFilter(threshold=0.0)
    X, cols = filt.fit_transform(X)
    assert X.shape == (5, 3)
    assert list(cols) == ["F2", "F3", "F4"]
    assert "F1" not in cols


def test_near_zero_variance_filter_no_features_removed():
    loader = DataLoader("project/tests/data/features/002_input.csv")
    X, _ = loader.load()
    filt = NearZeroVarianceFilter(threshold=0.0)
    X, cols = filt.fit_transform(X)
    assert X.shape == (5, 4)
    assert list(cols) == ["F1", "F2", "F3", "F4"]
