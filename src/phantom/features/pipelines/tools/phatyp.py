"""
Per-tool feature pipeline for PhaTYP (lifestyle prediction) output:
cleaning/filtering (used by the preprocessing step) and feature-matrix
construction (used by the extraction step).
"""

import pandas as pd


class PhatypFeaturePipeline:
    # Derives its feature matrix from the fixed TYPE column; needs no
    # interactively chosen column (see build_feature_matrix).
    NEEDS_FEATURE_COLUMN = False

    def __init__(self, min_phatyp_score: float = 0.9):
        self.min_phatyp_score = min_phatyp_score

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["PhaTYPScore"] = pd.to_numeric(df["PhaTYPScore"], errors="coerce")
        df = df[df["PhaTYPScore"] >= self.min_phatyp_score]
        return df[['Accession', 'TYPE']]

    def build_feature_matrix(self, df: pd.DataFrame, feature_col: str | None = None) -> pd.DataFrame:
        # feature_col is unused: TYPE is the only feature axis, not
        # interactively chosen. Kept for a uniform pipeline interface.
        df = df.copy()
        df["id"] = df["Accession"].str.split("_").str[0]
        counts = df.groupby(["id", "TYPE"]).size().unstack(fill_value=0)
        ratios = counts.div(counts.sum(axis=1), axis=0).round(4)
        ratios.columns = [f"{c.lower()}_ratio" for c in ratios.columns]
        return ratios.reset_index()
