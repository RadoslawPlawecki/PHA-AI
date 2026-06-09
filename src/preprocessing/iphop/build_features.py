"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
from typing import Optional
from ..normalization import apply_normalization
from ..features import calculate_predation_pressure, build_matrix


def build_features(df: pd.DataFrame, min_patients: int, feature_method: str, norm_method: str) -> pd.DataFrame:
    df = _filter_accessions(df)
    df = df.copy()
    df["id"] = df["Accession"].str.split("_").str[0]
    raw_matrix = _build_feature_matrix(df=df, min_patients=min_patients, method=feature_method)
    return apply_normalization(raw_matrix, method=norm_method)


def _filter_accessions(df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"^[^|]+\|S\d+_"
    return df[df["Accession"].str.match(pattern, na=False)].copy()


def _build_feature_matrix(df: pd.DataFrame, min_patients: int, method: str):
    if method.startswith("1") or method == "pp":
        return calculate_predation_pressure(df=df, min_patients=min_patients)
    return build_matrix(df=df, feature_col="genus", id_col="id", binary=True, min_patients=min_patients)
    