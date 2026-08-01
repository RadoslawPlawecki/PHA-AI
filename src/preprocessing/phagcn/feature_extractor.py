"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .build_features import build_features
from ..utils import format_accession, apply_mask, load_file
from ..cli import ask_column, ask_mask_file


class PhagcnFeatureExtractor:
    def __init__(self, min_phagcn_score: float = 0.5, min_patients: int = 4):
        self.min_phagcn_score = min_phagcn_score
        self.min_patients = max(1, min_patients)

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df = df[df["Prokaryotic virus (Bacteriophages and Archaeal virus)"] == "Y"]
        df = df[df["GenusCluster"] == "known_genus"]
        def extract_taxonomy(row):
            lineage_parts = row["Lineage"].split(";")
            scores = [float(x) for x in row["PhaGCNScore"].split(";")]
            taxonomy = {}
            for lineage, score in zip(lineage_parts, scores):
                rank, value = lineage.split(":", 1)
                taxonomy[rank] = (
                    value
                    if score >= self.min_phagcn_score
                    else None
                )
            return pd.Series(taxonomy)
        taxonomy_df = df.apply(extract_taxonomy, axis=1)
        df = pd.concat([df, taxonomy_df], axis=1)
        df = df[['Accession', 'genus']]
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = load_file("Accession", in_root)
        df = df.copy()
        mask_path = ask_mask_file(in_root)
        df = apply_mask(df, mask_path)
        filtered_df = self.preprocess(df, out_path=f"data/modalities/2.0/preprocessed/phagcn/{in_root.stem[:3]}_ChV_PGN_M_PP.csv")
        final_df = self._get_feat(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_PGN_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        col = ask_column(df)
        return build_features(df=df, feature_col=col, min_patients=self.min_patients)
        