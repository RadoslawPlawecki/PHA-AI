import pandas as pd
from pathlib import Path


def format_accession(prefix, column):
    return prefix + "|" + column.str.replace(r'k_', 'k', regex=True)

# ----- PHAVIP FILES PREPROCESSING
in_root = Path("data/phabox2/raw-merged/phavip")
out_root = Path("data/phabox2/preprocessed/phavip")

for file in in_root.iterdir():
    df = pd.read_csv(file, delimiter=';')
    gtool_id = parts = file.stem.split('_')[0]
    df["Genome"] = format_accession(gtool_id, df["Genome"])
    df["ORF"] = format_accession(gtool_id, df["ORF"])
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df = df[df["coverage"] >= 0.9]
    df = df[df["Annotation"] != "hypothetical protein"]

    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{file.stem}_PP.csv"
    df.to_csv(out_path, sep=';', index=False)


# ----- PHAGCN FILES PREPROCESSING
"""in_root = Path("data/phabox2/raw-merged/phagcn")
out_root = Path("data/phabox2/preprocessed/phagcn")

for file in in_root.iterdir():
    df = pd.read_csv(file, delimiter=';')
    gtool_id = parts = file.stem.split('_')[0]
    df["Accession"] = format_accession(gtool_id, df["Accession"])
    df = df[df["Prokaryotic virus (Bacteriophages and Archaeal virus)"] == "Y"]
    df = df[
        df["PhaGCNScore"]
        .str.split(";")
        .str[1]
        .astype(float) >= 0.9
    ]

    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{file.stem}_PP.csv"
    df.to_csv(out_path, sep=';', index=False)"""


# ----- CHERRY FILES PREPROCESSING
"""in_root = Path("data/phabox2/raw-merged/cherry")
out_root = Path("data/phabox2/preprocessed/cherry")

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

    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{file.stem}_PP.csv"
    df.to_csv(out_path, sep=';', index=False)"""
