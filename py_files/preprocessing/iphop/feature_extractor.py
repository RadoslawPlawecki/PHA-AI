"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
import questionary
from pathlib import Path
from typing import Optional

class IphopFeatureExtractor:
    def __init__(self, min_patients=2):
        if min_patients > 0:
            self.min_patients = min_patients
        
    def load_file(self, path: Path) -> pd.DataFrame:
        from ..utils import format_accession
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn')
        gtool_id = path.stem.split('_')[0]
        df["Accession"] = format_accession(gtool_id, df["Virus"])
        return df

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        from ..utils import split_taxonomy
        df = df.copy()
        df = split_taxonomy(df=df, col="Host genus", rename=True)
        virus_db = self._get_virus_db()
        df = df.merge(virus_db[['Accession', 'virus_genus']], on='Accession', how='left')
        df = df[['Accession', 'virus_genus', 'genus', 'Confidence score']]
        if out_path:
            df.to_csv(out_path, sep=';', index=False)
        return df

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = self.load_file(in_root)
        df = df.copy()
        filtered_df = self.preprocess(df, out_path=f"data/modalities/preprocessed/iphop/{in_root.stem[:3]}_ChV_IPH_M_PP.csv")
        final_df = self._get_feat(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_IPH_FEAT.csv"
        final_df.to_csv(out_path, sep=';', index=False)
        return final_df

    def _get_feat(self, df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
        df = df[df['Accession'].str.match(r'^[^|]+\|S\d+_', na=False)].copy()
        df['id'] = df['Accession'].str.split('_').str[0]
        if col is None:
            selectable_columns = [col for col in df.columns if col not in {'Accession', 'id'}]
            if not selectable_columns:
                raise ValueError("No valid columns to choose from.")
            col = questionary.select(
                "Choose a column:",
                choices=selectable_columns
            ).ask()
            print(f"\n[INFO] Selected column: {col}")
        return self._build_matrix(df, feature_col=col, id_col='id', binary=True)

    def _build_matrix(self, df: pd.DataFrame, feature_col: str, id_col: str, binary: bool = True) -> pd.DataFrame:
        matrix = pd.crosstab(df[feature_col], df[id_col])
        if binary:
            matrix = (matrix > 0).astype(int)
        vc_mask = matrix.sum(axis=0) >= self.min_patients
        matrix = matrix.loc[:, vc_mask].T
        return matrix.reset_index()

    @staticmethod
    def _get_virus_db():
        in_root = Path("data/modalities/preprocessed/phagcn")
        dfs = []
        for file in in_root.iterdir():
            if file.suffix == ".csv":
                df = pd.read_csv(file, delimiter=";")
                dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.rename(columns={"genus": "virus_genus"}, inplace=True)
        return combined_df

        