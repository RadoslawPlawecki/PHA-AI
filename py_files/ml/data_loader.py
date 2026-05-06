"""
@author: Radosław Pławecki
"""

import pandas as pd


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        df.set_index('id', inplace=True)

        y = (df.index.str.replace('S', '').astype(int) <= 34).astype(int)
        X = df.drop(columns=['label'], errors='ignore')

        return X.values, y, X.columns
        