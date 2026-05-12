"""
@author: Radosław Pławecki
"""

import pandas as pd


class DataLoader:
    def __init__(self, path=None):
        self.path = path

    def process(self, df):
        df = df.copy()
        sample_ids = df.index.to_series().astype(str).str.extract(r'(\d+)')[0]
        sample_ids = pd.to_numeric(sample_ids, errors='coerce')
        y = (sample_ids <= 34).astype(int)
        X = df.drop(columns=['label'], errors='ignore')
        return X.values, y.values, X.columns
        