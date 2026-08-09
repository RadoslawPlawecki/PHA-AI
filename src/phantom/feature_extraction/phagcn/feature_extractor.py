"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .build_features import build_features
from phantom.feature_extraction.utils import format_accession, apply_mask, load_file
from phantom.cli.prompts import FeatureExtractionPrompts


class PhagcnFeatureExtractor:
    def __init__(self, min_phagcn_score: float = 0.5, min_patients: int = 4, binary: bool = True):
        self.min_phagcn_score = min_phagcn_score
        self.min_patients = max(1, min_patients)
        self.binary = binary

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
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_file: Path, preprocessed_out_path: Path,
                     features_out_path: Path, feature_col: str = "genus", 
                     mask_path: Optional[str] = None) -> pd.DataFrame:
        df = load_file(in_file)
        if mask_path:
            df = apply_mask(df, mask_path)
        filtered_df = self.preprocess(df, out_path=preprocessed_out_path)
        final_df = self._get_feat(filtered_df, feature_col)
        features_out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(features_out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
        return build_features(df=df, feature_col=feature_col,
                              min_patients=self.min_patients, 
                              binary=self.binary)
