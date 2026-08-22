"""
Per-tool feature pipeline for PhaGCN (taxonomic composition) output:
cleaning/filtering (used by the preprocessing step) and feature-matrix
construction (used by the extraction step).
"""

import pandas as pd

from ..matrix import build_taxonomy_matrix


class PhagcnFeaturePipeline:
    # Its feature matrix is built over an interactively chosen taxonomy
    # column (see build_feature_matrix).
    NEEDS_FEATURE_COLUMN = True
    # binary/min_patients for the feature matrix are also interactively
    # chosen (see build_feature_matrix), analogous to feature_col above.
    NEEDS_MATRIX_OPTIONS = True

    def __init__(self, min_phagcn_score: float = 0.5, min_patients: int = 4, binary: bool = True):
        self.min_phagcn_score = min_phagcn_score
        self.min_patients = max(1, min_patients)
        self.binary = binary

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["Prokaryotic virus (Bacteriophages and Archaeal virus)"] == "Y"]
        df = df[df["GenusCluster"] == "known_genus"]

        def extract_taxonomy(row):
            lineage_parts = row["Lineage"].split(";")
            scores = [float(x) for x in row["PhaGCNScore"].split(";")]
            taxonomy = {}
            for lineage, score in zip(lineage_parts, scores):
                rank, value = lineage.split(":", 1)
                taxonomy[rank] = value if score >= self.min_phagcn_score else None
            return pd.Series(taxonomy)

        taxonomy_df = df.apply(extract_taxonomy, axis=1)
        df = pd.concat([df, taxonomy_df], axis=1)
        return df[['Accession', 'genus']]

    def build_feature_matrix(
        self, df: pd.DataFrame, feature_col: str = "genus",
        binary: bool | None = None, min_patients: int | None = None
    ) -> pd.DataFrame:
        return build_taxonomy_matrix(
            df,
            feature_col=feature_col,
            min_patients=self.min_patients if min_patients is None else min_patients,
            binary=self.binary if binary is None else binary,
        )
