"""Builders for the small structured DataFrames the suite runs on.

Plain functions on purpose -- the test data is a handful of rows of known
shape, not ORM objects, so factory_boy/Faker would only add indirection and
non-determinism. Headers mirror the real ``../data/features/*/raw-merged``
files so preprocessing filters exercise realistic column names.
"""

from __future__ import annotations

import pandas as pd

# A 7-rank lineage string in the format phantom's split_taxonomy expects.
_BACTEROIDES = (
    "d__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;"
    "f__Bacteroidaceae;g__Bacteroides;s__Bacteroides fragilis"
)
_BACTEROIDES_GTDB = (
    "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;"
    "f__Bacteroidaceae;g__Bacteroides;s__Bacteroides fragilis"
)


def contig_feature_df(pairs, col: str = "taxon") -> pd.DataFrame:
    """``pairs``: list[(accession, feature_value)] -> a merged-style frame
    with an ``Accession`` column and one feature column named ``col``."""
    return pd.DataFrame({"Accession": [a for a, _ in pairs], col: [v for _, v in pairs]})


def cherry_raw_df(rows=None) -> pd.DataFrame:
    """Columns used by CherryFeaturePipeline.preprocess()."""
    rows = rows or [
        ("geN|S1_k149_1||full", 1.0, _BACTEROIDES, _BACTEROIDES_GTDB),
        ("geN|S1_k149_2||full", 0.95, _BACTEROIDES, _BACTEROIDES_GTDB),
        ("geN|S2_k149_3||full", 1.0, _BACTEROIDES, _BACTEROIDES_GTDB),
    ]
    return pd.DataFrame(
        rows,
        columns=["Accession", "CHERRYScore", "Host_NCBI_lineage", "Host_GTDB_lineage"],
    )


def phagcn_raw_df(rows=None) -> pd.DataFrame:
    """Columns used by PhagcnFeaturePipeline.preprocess()."""
    rows = rows or [
        (
            "geN|S1_k149_1",
            "Y",
            "known_genus",
            "family:Straboviridae;genus:Tequatrovirus",
            "0.99;0.95",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Accession",
            "Prokaryotic virus (Bacteriophages and Archaeal virus)",
            "GenusCluster",
            "Lineage",
            "PhaGCNScore",
        ],
    )


def phavip_raw_df(rows) -> pd.DataFrame:
    """``rows``: list[(accession, coverage, pident, annotation)]."""
    return pd.DataFrame(rows, columns=["Accession", "coverage", "pident", "Annotation"])


def extracted_matrix(rows, feature_cols) -> pd.DataFrame:
    """An ``id`` column plus numeric feature columns -- the shape DataLoader
    reads at the classification stage. ``rows``: list[(id, *values)]."""
    return pd.DataFrame(rows, columns=["id", *feature_cols])
