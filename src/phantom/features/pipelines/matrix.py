"""
Shared feature-matrix construction helpers used by the per-tool pipelines.
"""

import pandas as pd


def derive_patient_id(accession: pd.Series) -> pd.Series:
    return accession.str.split("_").str[0]


def build_matrix(
    df: pd.DataFrame, feature_col: str, binary: bool = True, min_patients: int = 2,
    feature_columns: list[str] | None = None
) -> pd.DataFrame:
        df["id"] = derive_patient_id(df["Accession"])
        matrix = pd.crosstab(df["id"], df[feature_col])
        if binary:
            matrix = (matrix > 0).astype(int)
        if feature_columns is not None:
            matrix = matrix.reindex(columns=feature_columns, fill_value=0)
        else:
            vc_mask = matrix.sum(axis=0) >= min_patients
            matrix = matrix.loc[:, vc_mask]
        matrix.columns.name = None
        return matrix.reset_index()


def filter_accessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only rows whose Accession carries a sample-id tag (e.g.
    "geN|S1_..."), which merged/preprocessed feature data is expected to have.
    """
    pattern = r"^[^|]+\|S\d+_"
    return df[df["Accession"].str.match(pattern, na=False)].copy()


def build_taxonomy_matrix(
    df: pd.DataFrame, feature_col: str, min_patients: int, binary: bool = True,
    feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Shared by cherry/phagcn: filters to well-formed accessions, derives the
    patient id from the Accession, and builds an occurrence/count matrix
    over a chosen taxonomy column. When feature_columns is given, the matrix
    is reindexed to exactly those columns (0-filled) instead of recomputing
    the min_patients cutoff -- used to align a test-fold matrix to the
    column set already fixed by a train fold.
    """
    df = filter_accessions(df)
    df["id"] = derive_patient_id(df["Accession"])
    return build_matrix(
        df=df, feature_col=feature_col, binary=binary, min_patients=min_patients,
        feature_columns=feature_columns
    )
