"""
@author: Radosław Pławecki
"""

import pandas as pd
from typing import Optional


def build_matrix(df: pd.DataFrame, feature_col: str, id_col: str, binary: bool = True, min_patients: int = 2) -> pd.DataFrame:
        matrix = pd.crosstab(df[id_col], df[feature_col])
        if binary:
            matrix = (matrix > 0).astype(int)
        vc_mask = matrix.sum(axis=0) >= min_patients
        matrix = matrix.loc[:, vc_mask]
        return matrix.reset_index()


def calculate_predation_pressure(df: pd.DataFrame, min_patients: int = 2) -> pd.DataFrame:
        df['norm_score'] = df['Confidence score'] / df.groupby('Accession')['Confidence score'].transform('sum')
        B = df.pivot_table(index='Accession', columns='genus', values='norm_score', aggfunc='sum', fill_value=0)
        A = pd.crosstab(df['id'], df['Accession'])
        A = (A > 0).astype(float)
        common_viruses = list(set(A.columns).intersection(set(B.index)))
        A = A[common_viruses]
        B = B.loc[common_viruses]
        C = A.dot(B)
        mask = (C > 0).sum(axis=0) >= min_patients
        C = C.loc[:, mask]
        return C.reset_index().rename(columns={'index': 'id', 'row_0': 'id'})