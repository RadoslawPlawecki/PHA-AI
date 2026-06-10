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
        with open(self.input_path, newline="") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(self.input_path, sep=dialect.delimiter)
        sample_ids = pd.to_numeric(df['id'].str.extract(r'\|S(\d+)')[0], errors='coerce')
        y = (sample_ids <= 34).to_numpy(dtype=int)
        X = df.drop(columns=['id', 'label'], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce")
        self._log_load(X=X, y=y)
        return X, y

    def _log_load(self, X: pd.DataFrame, y: np.ndarray) -> None:
        counts = Counter(y)
        total = len(y)
        msg1 = f"Dataset loaded: {self.input_path}"
        msg2 = f"Samples: {total}"
        msg3 = f"Features: {X.shape[1]}"
        msg4 = "Class distribution:"
        class_lines = []
        for cls, count in sorted(counts.items()):
            pct = 100 * count / total
            class_lines.append(f"       class {cls}: {count} ({pct:.1f}%)")
        msg6 = f"Missing values: {int(X.isna().sum().sum())}"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
            self.logger.info(msg3)
            self.logger.info(msg4)
            for line in class_lines:
                self.logger.info(line)
            self.logger.info(msg6)
        else:
            print(msg1)
            print(msg2)
            print(msg3)
            print(msg4)
            print("\n".join(class_lines))
            print(msg6)
            