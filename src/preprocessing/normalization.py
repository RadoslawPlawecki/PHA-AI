"""
@author: Radosław Pławecki
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply_normalization(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'id' column.")
    if method.startswith("4"):
        return df.copy()
    ids = df["id"].copy()
    X = df.drop(columns=["id"]).astype(float)
    X = _apply_method(X, method)
    result = pd.concat([ids, X], axis=1)
    return result


def _apply_method(X: pd.DataFrame, method: str) -> pd.DataFrame:
    if "TSS" in method:
        X = _tss(X)
    elif "CLR" in method:
        X = _clr(X)
    if "Z-score" in method:
        X = _z_score(X)
    return X


def _tss(X: pd.DataFrame) -> pd.DataFrame:
    row_sums = X.sum(axis=1)
    X_norm = X.div(row_sums.replace(0, 1), axis=0)
    return X_norm


def _clr(X: pd.DataFrame) -> pd.DataFrame:
    pseudocount = 1e-6
    X_pseudo = X.replace(0, pseudocount)

    log_X = np.log(X_pseudo)
    geom_means = np.exp(log_X.mean(axis=1))

    return np.log(X_pseudo.div(geom_means, axis=0))


def _z_score(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    