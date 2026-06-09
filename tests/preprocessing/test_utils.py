"""
@author: Radosław Pławecki
"""

import pandas as pd
import pytest

from preprocessing.utils import format_accession, split_taxonomy


def test_format_accession_basic():
    s = pd.Series(["k_123", "k_abc"])
    result = format_accession("PRE", s)
    expected = pd.Series(["PRE|k123", "PRE|kabc"])
    pd.testing.assert_series_equal(result, expected)


def test_format_accession_mixed_values():
    s = pd.Series(["k_1", "no_prefix", "k_test"])
    result = format_accession("X", s)
    expected = pd.Series(["X|k1", "X|no_prefix", "X|ktest"])
    pd.testing.assert_series_equal(result, expected)


def test_split_taxonomy_without_rename():
    df = pd.DataFrame({
        "lineage": [
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
            "d__Archaea;p__Euryarchaeota"
        ]
    })
    result = split_taxonomy(df, col="lineage", rename=False)
    assert "domain" not in result.columns
    assert result.loc[0, "d"] == "Bacteria"
    assert result.loc[0, "p"] == "Proteobacteria"
    assert result.loc[0, "c"] == "Gammaproteobacteria"


def test_split_taxonomy_with_rename():
    df = pd.DataFrame({
        "lineage": [
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales"
        ]
    })
    result = split_taxonomy(df, rename=True)
    assert "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria"  
    assert "domain" in result.columns
    assert "phylum" in result.columns
    assert "d" not in result.columns
    assert result.loc[0, "domain"] == "Bacteria"
    assert result.loc[0, "phylum"] == "Proteobacteria"


def test_split_taxonomy_with_prefix_rename():
    df = pd.DataFrame({
        "lineage": [
            "d__Bacteria;p__Proteobacteria"
        ]
    })
    result = split_taxonomy(df, rename=True, lineage_type="tax")
    assert "tax_domain" in result.columns
    assert "tax_phylum" in result.columns
    assert result.loc[0, "tax_domain"] == "Bacteria"


def test_split_taxonomy_missing_levels():
    df = pd.DataFrame({
        "lineage": ["d__Bacteria"]
    })
    result = split_taxonomy(df)
    assert result.loc[0, "domain"] == "Bacteria"
    assert "phylum" not in result.columns or pd.isna(result.loc[0, "phylum"])


def test_split_taxonomy_empty_values():
    df = pd.DataFrame({
        "lineage": [""]
    })
    result = split_taxonomy(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
