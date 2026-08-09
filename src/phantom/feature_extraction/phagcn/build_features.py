"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
from typing import Optional
from phantom.feature_extraction.features import build_matrix


def build_features(df: pd.DataFrame, feature_col: str, min_patients: int, binary: bool) -> pd.DataFrame:
    df = filter_accessions(df)
    df = df.copy()
    df["id"] = df["Accession"].str.split("_").str[0]
    feature_matrix = build_matrix(df=df, feature_col=feature_col, min_patients=min_patients, binary=binary)
    return feature_matrix


def filter_accessions(df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"^[^|]+\|S\d+_"
    return df[df["Accession"].str.match(pattern, na=False)].copy()
