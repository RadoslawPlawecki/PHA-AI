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

class IphopFeatureExtractor:
    def __init__(self, min_patients: int = 1):
        self.min_patients = max(1, min_patients)
        
    def load_file(self, path: Path) -> pd.DataFrame:
        from ..utils import format_accession
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn')
        gtool_id = path.stem.split('_')[0]
        df["Accession"] = format_accession(gtool_id, df["Virus"])
        return df

    def preprocess(self, df: pd.DataFrame, out_path: Optional[str] = None) -> pd.DataFrame:
        from ..utils import split_taxonomy
        df = df.copy()
        df = df[df["Confidence score"] >= 90]
        df = split_taxonomy(df=df, col="Host genus", rename=True)
        df = df[['Accession', 'genus', 'Confidence score']]
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
        method = questionary.select(
            "Select feature engineering method for the phage-host relation:",
            choices=[
                "1) Predation Pressure",
                "2) Occurrence Matrix"
            ]
        ).ask()
        print(f"\n[INFO] Selected feature engineering method: {method}")
        if method.startswith("1"):
            raw_matrix = self._calc_predation_pressure(df)
        else:
            raw_matrix = self._build_matrix(df, feature_col='genus', id_col='id', binary=True)
        norm_method = questionary.select(
            "Select normalization method:",
            choices=[
                "1) TSS + Z-score",
                "2) CLR + Z-score",
                "3) Only TSS",
                "4) Nothing (raw data)"
            ]
        ).ask()
        print(f"[INFO] Selected normalization: {norm_method}")
        return self._apply_normalization(raw_matrix, method=norm_method)

    def _calc_predation_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        df['norm_score'] = df['Confidence score'] / df.groupby('Accession')['Confidence score'].transform('sum')
        B = df.pivot_table(index='Accession', columns='genus', values='norm_score', aggfunc='sum', fill_value=0)
        A = pd.crosstab(df['id'], df['Accession'])
        A = (A > 0).astype(float)
        common_viruses = list(set(A.columns).intersection(set(B.index)))
        A = A[common_viruses]
        B = B.loc[common_viruses]
        C = A.dot(B)
        mask = (C > 0).sum(axis=0) >= self.min_patients
        C = C.loc[:, mask]
        return C.reset_index().rename(columns={'index': 'id', 'row_0': 'id'})

    def _build_matrix(self, df: pd.DataFrame, feature_col: str, id_col: str, binary: bool = True) -> pd.DataFrame:
        matrix = pd.crosstab(df[id_col], df[feature_col])
        if binary:
            matrix = (matrix > 0).astype(int)
        vc_mask = matrix.sum(axis=0) >= self.min_patients
        matrix = matrix.loc[:, vc_mask]
        return matrix.reset_index()

    def _apply_normalization(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        if method.startswith("4"):
            return df
        ids = df['id']
        X = df.drop(columns=['id']).astype(float)
        if "TSS" in method:  # Total Sum Scaling
            row_sums = X.sum(axis=1)
            X = X.div(row_sums.replace(0, 1), axis=0)
        elif "CLR" in method:  # Centered Log-Ratio
            pseudocount = 1e-6
            X_pseudo = X.replace(0, pseudocount)
            geom_means = np.exp(np.mean(np.log(X_pseudo), axis=1))
            X = np.log(X_pseudo.div(geom_means, axis=0))
        if "Z-score" in method:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        result = pd.concat([ids, X], axis=1)
        return result
        