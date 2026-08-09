"""
Manages the metadata file that stores information about samples and
dataset statistics.
"""

from pathlib import Path
import pandas as pd

class MetadataManager():
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["id"])
        return pd.read_csv(self.path, sep=";")

    def save(self, df: pd.DataFrame):
        df.to_csv(self.path, sep=";", index=False)

    def add_metadata_columns(self, new_df: pd.DataFrame):
        """
        Merges new metadata on 'id'. Fails if duplicate columns (other 
        than 'id') exist.
        """
        if new_df.empty or "id" not in new_df.columns:
            return
        current_df = self.load()
        if current_df.empty:
            self.save(new_df)
            return
        duplicate_cols = (
            set(new_df.columns).intersection(current_df.columns) - {"id"}
        )
        if duplicate_cols:
            raise ValueError(f"Duplicate columns found: {duplicate_cols}. Incrementing not allowed.")
        merged = pd.merge(current_df, new_df, on="id", how="outer")
        if "id" in merged.columns:
            merged["_sort_key"] = (
                merged["id"].str
                            .extract(r'(\d+)', expand=False)
                            .astype(float)
            )
            merged = (
                merged.sort_values("_sort_key")
                      .drop(columns=["_sort_key"])
                      .reset_index(drop=True)
            )
        self.save(merged)
        