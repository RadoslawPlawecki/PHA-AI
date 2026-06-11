"""
@author: Radosław Pławecki
"""

import numpy as np
import pytest
from ml.data_loader import DataLoader 


def test_data_loader_basic():
    loader = DataLoader("project/tests/data/features/001_input.csv")
    X, y, _ = loader.load()
    assert X.shape == (5, 4)
    assert len(y) == 5
    assert np.array_equal(y, np.array([1, 1, 0, 0, 1]))
    assert X.isna().sum().sum() == 0


def test_data_loader_invalid_path_raises():
    with pytest.raises(FileNotFoundError):
        DataLoader("this_file_does_not_exist.csv")
