"""
@author: Radosław Pławecki
"""

import pandas as pd
from pandas.testing import assert_frame_equal
from preprocessing.features import build_matrix, calculate_predation_pressure


def test_build_matrix_binary_true():
    df = pd.read_csv("project/tests/data/occurence_matrix/input.csv")
    expected = pd.read_csv("project/tests/data/occurence_matrix/expected_binary_true.csv")
    result = build_matrix(df, feature_col="feature", id_col="id", binary=True, min_patients=2)
    result = result.sort_values("id").reset_index(drop=True)
    expected = expected.sort_values("id").reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_build_matrix_binary_false():
    df = pd.read_csv("project/tests/data/occurence_matrix/input.csv")
    expected = pd.read_csv("project/tests/data/occurence_matrix/expected_binary_false.csv")
    result = build_matrix(df, feature_col="feature", id_col="id", binary=False, min_patients=2)
    result = result.sort_values("id").reset_index(drop=True)
    expected = expected.sort_values("id").reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_calculate_predation_pressure():
    df = pd.read_csv("project/tests/data/predation_pressure/input.csv")
    expected = pd.read_csv("project/tests/data/predation_pressure/expected.csv")
    result = calculate_predation_pressure(df)
    result = result.sort_values("id").reset_index(drop=True)
    expected = expected.sort_values("id").reset_index(drop=True)
    assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )
