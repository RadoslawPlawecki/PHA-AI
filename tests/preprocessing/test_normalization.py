"""
@author: Radosław Pławecki
"""

import pytest
import numpy as np
import pandas as pd
from preprocessing.normalization import apply_normalization

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "A": [1.0, 2.0, 3.0],
        "B": [4.0, 5.0, 6.0],
    })


def test_missing_id_raises():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError, match="id"):
        apply_normalization(df, "TSS")


def test_method_4_returns_copy(sample_df):
    result = apply_normalization(sample_df, "4-anything")
    pd.testing.assert_frame_equal(result, sample_df)
    assert result is not sample_df

def test_tss_normalization(sample_df):
    result = apply_normalization(sample_df, "TSS")
    assert "id" in result.columns
    row_sums = result[["A", "B"]].sum(axis=1).round(6)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0])


def test_tss_handles_zero_row():
    df = pd.DataFrame({
        "id": [1, 2],
        "A": [0.0, 1.0],
        "B": [0.0, 1.0],
    })
    result = apply_normalization(df, "TSS")
    assert not result[["A", "B"]].isna().any().any()


def test_clr_no_zero_handling():
    df = pd.DataFrame({
        "id": [1],
        "A": [1.0],
        "B": [2.0],
    })
    result = apply_normalization(df, "CLR")
    assert np.isfinite(result[["A", "B"]].values).all()


def test_clr_zero_replacement():
    df = pd.DataFrame({
        "id": [1],
        "A": [0.0],
        "B": [2.0],
    })
    result = apply_normalization(df, "CLR")
    assert not (result[["A", "B"]] == 0).any().any()


def test_z_score_mean_std(sample_df):
    result = apply_normalization(sample_df, "Z-score")
    X = result[["A", "B"]]
    np.testing.assert_allclose(X.mean().values, [0, 0], atol=1e-6)
    np.testing.assert_allclose(X.std(ddof=0).values, [1, 1], atol=1e-6)


def test_z_score_preserves_shape(sample_df):
    result = apply_normalization(sample_df, "Z-score")
    assert result.shape == sample_df.shape
    assert list(result.columns) == list(sample_df.columns)

def test_tss_then_zscore(sample_df):
    result = apply_normalization(sample_df, "TSS-Z-score")
    X = result[["A", "B"]]
    assert np.isfinite(X.values).all()
    assert X.shape == (3, 2)
