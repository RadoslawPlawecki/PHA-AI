"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path


def format_accession(prefix, column):
    return prefix + "|" + column.str.replace(r'k_', 'k', regex=True)


def split_taxonomy(df, col="lineage", rename=True, lineage_type=None):
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


# ----- PHAGCN FILES PREPROCESSING
"""in_root = Path("data/modalities/raw-merged/phagcn")
out_root = Path("data/modalities/preprocessed/phagcn")

for file in in_root.iterdir():
    df = pd.read_csv(file, delimiter=';')
    gtool_id = parts = file.stem.split('_')[0]
    df["Accession"] = format_accession(gtool_id, df["Accession"])
    df = df[df["Prokaryotic virus (Bacteriophages and Archaeal virus)"] == "Y"]
    df = df[df["GenusCluster"] == "known_genus"]
    df = df[
        df["PhaGCNScore"]
        .str.split(";")
        .str[1]
        .astype(float) >= 0.9
    ]
    split_lineage = (
        df["Lineage"]
        .str.split(";")
        .explode()
        .str.split(":", n=1, expand=True)
        .pivot(columns=0, values=1)
    )
    df = pd.concat([df, split_lineage], axis=1)
    df = df[['Accession', 'genus', 'species']]
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{file.stem}_PP.csv"
    df.to_csv(out_path, sep=';', index=False)"""


# ----- CHERRY FILES PREPROCESSING
"""in_root = Path("data/modalities/raw-merged/cherry")
out_root = Path("data/modalities/preprocessed/cherry")

for file in in_root.iterdir():
    df = pd.read_csv(file, delimiter=';')
    gtool_id = parts = file.stem.split('_')[0]
    df["Accession"] = format_accession(gtool_id, df["Accession"])
    df["CHERRYScore"] = pd.to_numeric(df["CHERRYScore"], errors="coerce")
    df = df[df["CHERRYScore"] >= 0.9]
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
    for col in df.columns:
        print(df[col].value_counts())
    df = df[['Accession', 'Host', 'ncbi_phylum',
            'ncbi_class', 'ncbi_order', 'ncbi_family', 
            'ncbi_genus', 'ncbi_species', 'gtdb_phylum',
            'gtdb_class', 'gtdb_order', 'gtdb_family',
            'gtdb_genus', 'gtdb_species']]
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{file.stem}_PP.csv"
    df.to_csv(out_path, sep=';', index=False)"""
