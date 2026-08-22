"""
Per-tool feature pipeline for CHERRY (phage-host prediction) output:
cleaning/filtering (used by the preprocessing step) and feature-matrix
construction (used by the extraction step).
"""

import pandas as pd

from ..matrix import build_taxonomy_matrix
from ..utils import split_taxonomy


class CherryFeaturePipeline:
    # Its feature matrix is built over an interactively chosen taxonomy
    # column (see build_feature_matrix).
    NEEDS_FEATURE_COLUMN = True
    # binary/min_patients for the feature matrix are also interactively
    # chosen (see build_feature_matrix), analogous to feature_col above.
    NEEDS_MATRIX_OPTIONS = True

    def __init__(self, min_cherry_score: float = 0.9, min_patients: int = 10, binary: bool = True):
        self.min_cherry_score = min_cherry_score
        self.min_patients = max(1, min_patients)
        self.binary = binary

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["CHERRYScore"] = pd.to_numeric(df["CHERRYScore"], errors="coerce")
        df = df[df["CHERRYScore"] >= self.min_cherry_score]
        df = df[
            ~(
                (df["Host_NCBI_lineage"] == "-") &
                (df["Host_GTDB_lineage"].isin(["-", "Not found"]))
            )
        ]
        df = split_taxonomy(df, col="Host_NCBI_lineage", lineage_type='ncbi')
        df = split_taxonomy(df, col="Host_GTDB_lineage", lineage_type='gtdb')
        df = df[df["ncbi_domain"] == "Bacteria"]
        df = df[df["ncbi_genus"] != "NAmissing"]
        df = df[df["ncbi_species"] != "bacterium"]
        return df[['Accession', 'ncbi_phylum', 'ncbi_class', 'ncbi_order',
                    'ncbi_family', 'ncbi_genus', 'ncbi_species']]

    def build_feature_matrix(
        self, df: pd.DataFrame, feature_col: str,
        binary: bool | None = None, min_patients: int | None = None
    ) -> pd.DataFrame:
        return build_taxonomy_matrix(
            df,
            feature_col=feature_col,
            min_patients=self.min_patients if min_patients is None else min_patients,
            binary=self.binary if binary is None else binary,
        )
