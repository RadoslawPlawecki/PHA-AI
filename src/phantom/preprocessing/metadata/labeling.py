"""
Handles data labeling using various strategies, including manual labels, 
filename patterns, sample thresholds and CSV-based mappings.
"""

from abc import ABC, abstractmethod
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

class LabelingStrategy(ABC):
    @abstractmethod
    def assign_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        pass

class ManualLabeler(LabelingStrategy):
    def __init__(self, positive_files: List[str]):
        self.positive_files = set(positive_files)

    def assign_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        return {
            row["id"]: (1 if row["id"] in self.positive_files else 0)
            for _, row in df.iterrows()
        }

class PatternLabeler(LabelingStrategy):
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def assign_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        return {
            row["id"]: (1 if self.pattern.search(str(row["id"])) else 0)
            for _, row in df.iterrows()
        }

class SampleThresholdLabeler(LabelingStrategy):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def assign_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        labels = {}
        for _, row in df.iterrows():
            match = re.search(r"\d+", str(row["id"]))
            sample_num = int(match.group()) if match else float('inf')
            labels[row["id"]] = (
                1 if sample_num <= self.threshold else 0
            )
        return labels

class ExternalCSVLabeler(LabelingStrategy):
    def __init__(self, csv_path: Path, join_key: str = "original_file", 
                 label_col: str = "label"):
        self.csv_path = Path(csv_path)
        self.join_key = join_key
        self.label_col = label_col

    def assign_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        external_df = pd.read_csv(self.csv_path)
        merged = df.merge(external_df, on=self.join_key, how="left")
        return dict(zip(merged["id"], 
                        merged[self.label_col].fillna(0).astype(int)))
    