"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np
import csv
from collections import Counter
from pathlib import Path


class DataLoader:
    def __init__(self, input_path: str, logger=None):
        self.input_path = Path(input_path)
        self.logger = logger
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def load(self):
        try:
            with open(self.input_path, newline="") as f:
                sample = f.read(4096)
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
        except:
            sep = ";"
        df = pd.read_csv(self.input_path, sep=sep)
        sample_ids = df['id'].copy()
        y = (
            pd.to_numeric(
                sample_ids.str.extract(r'\|S(\d+)')[0],
                errors='coerce'
            ) <= 34
        ).to_numpy(dtype=int)
        X = df.drop(columns=['id', 'label'], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce")
        self._log_load(X=X, y=y)
        return X, y, sample_ids

    def _log_load(self, X: pd.DataFrame, y: np.ndarray) -> None:
        counts = Counter(y)
        total = len(y)
        msg1 = f"Dataset Loaded: {self.input_path}"
        msg2 = f"Samples: {total}"
        msg3 = f"Features: {X.shape[1]}"
        class_distribution = "Class Distribution:"
        class_lines = []
        for cls, count in sorted(counts.items()):
            pct = 100 * count / total
            class_lines.append(f"       class {cls}: {count} ({pct:.1f}%)")
        msg4 = f"\n{class_distribution}\n{"\n".join(class_lines)}\n"
        msg5 = f"Missing Values: {int(X.isna().sum().sum())}"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
            self.logger.info(msg3)
            self.logger.info(msg4)
            self.logger.info(msg5)
        else:
            print(msg1)
            print(msg2)
            print(msg3)
            print(msg4)
            print(msg5)
            