"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils import apply_mask, load_file
from ..cli import ask_mask_file


class PhatypFeatureExtractor:
    def __init__(self, min_phatyp_score: float = 0.9):
        self.min_phatyp_score = min_phatyp_score

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df["PhaTYPScore"] = pd.to_numeric(df["PhaTYPScore"], errors="coerce")
        df = df[df["PhaTYPScore"] >= self.min_phatyp_score]
        df = df[['Accession', 'TYPE']]
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def calculate_type_ratios(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> pd.DataFrame:
        df = df.copy()
        df["id"] = df["Accession"].str.split("_").str[0]
        counts = df.groupby(["id", "TYPE"]).size().unstack(fill_value=0)
        ratios = counts.div(counts.sum(axis=1), axis=0).round(4)
        ratios.columns = [
            f"{c.lower()}_ratio"
            for c in ratios.columns
        ]
        result = ratios.reset_index()
        return result

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = load_file(in_root)
        df = df.copy()
        mask_path = ask_mask_file(in_root)
        df = apply_mask(df, mask_path)
        filtered_df = self.preprocess(df, out_path=f"data/modalities/2.0/preprocessed/phatyp/{in_root.stem[:3]}_ChV_PHT_M_PP.csv")
        final_df = self.calculate_type_ratios(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_PHT_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df
        