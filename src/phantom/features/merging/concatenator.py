"""
Handles merging raw per-sample feature files (TSV) into combined dataset 
files.
"""

import os
from pathlib import Path
import pandas as pd
from phantom.cli.features import FeatureCollectionPrompts
from phantom.config.features import FeatureConfigManager


class FeatureConcatenator:
    def __init__(self, version: str, config_mgr: FeatureConfigManager | None = None):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.raw_dir = self.config_mgr.get_stage_dir(version, "raw")
        self.raw_merged_dir = self.config_mgr.get_stage_dir(version, "raw_merged")

    def run(self) -> None:
        print(f"\n[INFO] Starting feature merging for version {self.version}...")
        self.merge_raw_files()

    def merge_raw_files(self) -> None:
        if not self.raw_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.raw_dir}")
            return
        leaf_dirs = self._find_leaf_dirs(self.raw_dir)
        if not leaf_dirs:
            print(f"[ERROR] No .tsv/.csv files found anywhere under {self.raw_dir}")
            return
        for leaf_dir in leaf_dirs:
            self._process_leaf_directory(leaf_dir)

    def _find_leaf_dirs(self, root: Path) -> list[Path]:
        """
        Recursively finds every directory that directly contains .tsv/.csv
        files, regardless of how deeply it is nested under `root`. This
        makes merging independent of the exact directory depth produced
        during import (a single "<tool>/<module>" level is the common
        case, but a directory of raw files at any depth must still merge).
        """
        leaf_dirs = []
        for dirpath, _, filenames in os.walk(root):
            if any(f.endswith((".tsv", ".csv")) for f in filenames):
                leaf_dirs.append(Path(dirpath))
        return sorted(leaf_dirs)

    def _process_leaf_directory(self, leaf_dir: Path) -> None:
        raw_files = sorted(
            list(leaf_dir.glob("*.tsv")) + list(leaf_dir.glob("*.csv"))
        )
        if not raw_files:
            return
        rel_dir = leaf_dir.relative_to(self.raw_dir)
        folder_label = str(rel_dir)
        id_col = self._get_id_column(raw_files[0], folder_label)
        if not id_col:
            print(f"[ERROR] Skipped {folder_label}: No ID column selected.")
            return
        dfs = []
        sample_name = None
        for file in raw_files:
            df, s_name = self._read_and_format_file(file, id_col)
            if df is not None:
                dfs.append(df)
                if sample_name is None:
                    sample_name = s_name
        if dfs:
            self._save_merged_df(dfs, rel_dir, sample_name)

    def _get_id_column(self, sample_file: Path, folder_label: str) -> str | None:
        sep = "\t" if sample_file.suffix.lower() == ".tsv" else ";"
        try:
            peek_df = pd.read_csv(sample_file, sep=sep, nrows=2)
        except Exception:
            peek_df = pd.read_csv(sample_file, sep=None, engine="python", nrows=2)
        return FeatureCollectionPrompts.ask_id_column(folder_label, list(peek_df.columns))

    def _read_and_format_file(
        self, file: Path, id_col: str
    ) -> tuple[pd.DataFrame | None, str | None]:
        parts = file.stem.split("_")
        sample_id = parts[0]
        sample_name = "_".join(parts[1:]) if len(parts) > 1 else None
        sep = "\t" if file.suffix.lower() == ".tsv" else ";"
        try:
            df = pd.read_csv(file, sep=sep)
        except Exception:
            df = pd.read_csv(file, sep=None, engine="python")
        if id_col not in df.columns:
            print(f"[ERROR] Warning: Column '{id_col}' not found in {file.name}. Skipping file.")
            return None, None
        df[id_col] = sample_id + "_" + df[id_col].astype(str)
        if id_col != "Accession":
            df = df.rename(columns={id_col: "Accession"})
        return df, sample_name

    def _save_merged_df(
        self,
        dfs: list[pd.DataFrame],
        rel_dir: Path,
        sample_name: str | None,
    ) -> None:
        merged = pd.concat(dfs, ignore_index=True)
        out_dir = self.raw_merged_dir / rel_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{sample_name or rel_dir.name}_M.csv"
        out_path = out_dir / out_filename
        merged.to_csv(out_path, sep=";", index=False)
        print(f"[INFO] Merged: {rel_dir}")
        print(f"       Saved to: {out_path} (Shape: {merged.shape})")
