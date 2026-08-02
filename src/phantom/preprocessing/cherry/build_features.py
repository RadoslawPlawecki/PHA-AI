"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
from typing import Optional
from phantom.preprocessing.features import build_matrix


def build_features(df: pd.DataFrame, min_patients: int, feature_col: str) -> pd.DataFrame:
    df = _filter_accessions(df)
    df = df.copy()
    df["id"] = df["Accession"].str.split("_").str[0]
    feature_matrix = _build_feature_matrix(df=df, min_patients=min_patients, feature_col=feature_col)
    return feature_matrix


def _filter_accessions(df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"^[^|]+\|S\d+_"
    return df[df["Accession"].str.match(pattern, na=False)].copy()


def _build_feature_matrix(df: pd.DataFrame, min_patients: int, feature_col: str):
    return build_matrix(df=df, feature_col=feature_col, binary=True, min_patients=min_patients)
