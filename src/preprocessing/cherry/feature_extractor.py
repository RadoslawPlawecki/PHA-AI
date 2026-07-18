"""
@author: Radosław Pławecki
"""

import pandas as pd
import questionary
from pathlib import Path
from typing import Optional
from .build_features import build_features
from ..utils import format_accession, split_taxonomy, apply_mask
from ..cli import ask_column, ask_mask_file


class CherryFeatureExtractor:
    def __init__(self, min_cherry_score: float = 0.9, min_patients: int = 10):
        self.min_cherry_score = min_cherry_score
        self.min_patients = max(1, min_patients)
        
    def load_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn')
        gtool_id = path.stem.split('_')[0]
        df["Accession"] = format_accession(gtool_id, df["Accession"])
        return df

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
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
        df = df[['Accession', 'ncbi_genus', 'ncbi_species', 'gtdb_genus', 'gtdb_species']]
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = self.load_file(in_root)
        df = df.copy()
        mask_path = ask_mask_file(in_root)
        df = apply_mask(df, mask_path)
        filtered_df = self.preprocess(df, out_path=f"data/modalities/2.0/preprocessed/cherry/{in_root.stem[:3]}_ChV_CHR_M_PP.csv")
        final_df = self._get_feat(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_CHR_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        col = ask_column(df)
        return build_features(df=df, feature_col=col, min_patients=self.min_patients)
        