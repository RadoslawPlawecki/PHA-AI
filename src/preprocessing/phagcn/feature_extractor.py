"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
import questionary
from pathlib import Path
from typing import Optional
from .build_features import build_features
from ..utils import format_accession, split_taxonomy
from .cli import ask_column


class PhagcnFeatureExtractor:
    def __init__(self, min_phagcn_score: float = 0.9, min_patients: int = 2):
        self.min_phagcn_score = min_phagcn_score
        self.min_patients = max(1, min_patients)
        
    def load_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn')
        gtool_id = path.stem.split('_')[0]
        df["Accession"] = format_accession(gtool_id, df["Accession"])
        return df

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df = df[df["Prokaryotic virus (Bacteriophages and Archaeal virus)"] == "Y"]
        df = df[df["GenusCluster"] == "known_genus"]
        df = df[
            df["PhaGCNScore"]
            .str.split(";")
            .str[1]
            .astype(float) >= self.min_phagcn_score
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
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = self.load_file(in_root)
        df = df.copy()
        filtered_df = self.preprocess(df, out_path=f"data/modalities/preprocessed/phagcn/{in_root.stem[:3]}_ChV_PGN_M_PP.csv")
        final_df = self._get_feat(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_PGN_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        col = ask_column(df)
        return build_features(df=df, feature_col=col, min_patients=self.min_patients)
        