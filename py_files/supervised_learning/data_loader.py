"""
@author: Radosław Pławecki
"""

import pandas as pd


class DataLoader:
    def __init__(self, path=None):
        self.path = path

    def process(self, df):
        df = df.copy()
        y = (df.index.str.replace('S', '').astype(int) <= 34).astype(int)
        X = df.drop(columns=['label'], errors='ignore')
        return X.values, y, X.columns
        