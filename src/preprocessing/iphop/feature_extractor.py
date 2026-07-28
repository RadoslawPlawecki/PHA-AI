"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
import questionary
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from ..cli import ask_feature_method, ask_normalization_method, ask_mask_file
from .build_features import build_features
from ..utils import split_taxonomy, apply_mask, load_file
from ..normalization import apply_normalization


class IphopFeatureExtractor:
    def __init__(self, min_patients: int = 1):
        self.min_patients = max(1, min_patients)

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df = df[df["Confidence score"] >= 90]
        df = split_taxonomy(df=df, col="Host genus", rename=True)
        df = df[['Accession', 'genus', 'Confidence score']]
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = load_file("Virus", in_root)
        df = df.copy()
        mask_path = ask_mask_file(in_root)
        df = apply_mask(df, mask_path)
        filtered_df = self.preprocess(df, out_path=f"data/modalities/2.0/preprocessed/iphop/{in_root.stem[:3]}_ChV_IPH_M_PP.csv")
        final_df = self._get_feat(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_IPH_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
        feature_method = ask_feature_method()
        if not feature_method.startswith("1"):
            norm_method = "4) Nothing (raw data) [raw]"  # no normalization, if occurence matrix as a feature
        else:
            norm_method = ask_normalization_method()
        return build_features(df=df, min_patients=self.min_patients, feature_method=feature_method, norm_method=norm_method)
        