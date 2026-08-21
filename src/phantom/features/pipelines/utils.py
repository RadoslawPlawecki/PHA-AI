"""
Shared IO helpers used when loading and masking raw-merged feature data
before it is handed to a per-tool pipeline.
"""

import pandas as pd
from typing import Optional
from pathlib import Path


def format_accession(prefix: str, column: pd.Series) -> pd.Series:
        return prefix + "|" + column.str.replace(r'k_', 'k', regex=False)


def split_taxonomy(df: pd.DataFrame, col: str = "lineage", rename: bool = True, lineage_type: Optional[str] = None):
    if lineage_type:
        rename_map = {
            "d": f"{lineage_type}_domain",
            "p": f"{lineage_type}_phylum",
            "c": f"{lineage_type}_class",
            "o": f"{lineage_type}_order",
            "f": f"{lineage_type}_family",
            "g": f"{lineage_type}_genus",
            "s": f"{lineage_type}_species"
        }
    else:
        rename_map = {
        "d": "domain",
        "p": "phylum",
        "c": "class",
        "o": "order",
        "f": "family",
        "g": "genus",
        "s": "species"
    }
    expanded = df[col].apply(
        lambda x: dict(item.split("__", 1) for item in x.split(";") if "__" in item)
    ).apply(pd.Series)
    if rename:
        expanded = expanded.rename(columns=rename_map)
    return pd.concat([df, expanded], axis=1)


def apply_mask(df: pd.DataFrame, mask_path: Path) -> pd.DataFrame:
    if mask_path is None:
        print(f"[INFO] No mask applied. Records retained: {len(df)}")
        return df
    mask_df = pd.read_csv(mask_path, sep="\t")
    if "contig_id" not in mask_df.columns:
        raise ValueError("Mask file must contain a 'contig_id' column.")
    original_count = len(df)
    mask_contigs = (
        mask_df["contig_id"]
        .str.split(":", n=1)
        .str[0]
        .unique()
    )
    df_contigs = (
        df["Accession"]
        .str.split(":", n=1)
        .str[0]
    )
    df = df[df_contigs.isin(mask_contigs)]
    print(
        f"[INFO] Mask applied: {mask_path.name}\n"
        f"       Retrieved {len(df)} / {original_count} records\n"
        f"       Unique contigs retained: {df_contigs.isin(mask_contigs).sum()}"
    )
    return df


def load_file(path: Path, accession: str = "Accession") -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";", on_bad_lines="warn")
    vt = path.stem.split("_")[0]
    df["Accession"] = format_accession(vt, df[accession])
    return df
