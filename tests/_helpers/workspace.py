"""A throwaway on-disk data workspace for integration / controller tests.

Lays out the sibling ``data/`` tree the pipeline expects (feature version
9.9 with its four stage directories, plus per-vtool mask directories) next
to a writable ``config.toml``. Build one via the ``tmp_workspace`` fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VTOOLS = {"geN": "genomad", "VIB": "vibrant", "VS2": "virsorter2"}
TOOLS = ("cherry", "phagcn", "phavip", "phatyp")
STAGE_DIRS = {
    "raw": "raw",
    "raw_merged": "raw-merged",
    "preprocessed": "preprocessed",
    "extracted": "extracted",
}


@dataclass
class Workspace:
    root: Path
    config_path: Path
    version: str = "9.9"

    @property
    def data_root(self) -> Path:
        return self.root / "data"

    @property
    def features_root(self) -> Path:
        return self.data_root / "features" / self.version

    def stage_dir(self, stage: str, tool: str | None = None) -> Path:
        d = self.features_root / STAGE_DIRS[stage]
        return d / tool if tool else d

    def write_raw_merged(self, tool: str, filename: str, df: pd.DataFrame) -> Path:
        return self._write_csv(self.stage_dir("raw_merged", tool), filename, df)

    def write_preprocessed(self, tool: str, filename: str, df: pd.DataFrame) -> Path:
        return self._write_csv(self.stage_dir("preprocessed", tool), filename, df)

    def write_extracted(self, tool: str, filename: str, df: pd.DataFrame) -> Path:
        return self._write_csv(self.stage_dir("extracted", tool), filename, df)

    def write_raw(self, rel_dir: str, filename: str, df: pd.DataFrame, sep: str = "\t") -> Path:
        target = self.stage_dir("raw") / rel_dir
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        df.to_csv(path, sep=sep, index=False)
        return path

    def write_mask(self, vtool_name: str, filename: str, rows: list[dict]) -> Path:
        target = self.data_root / "masks" / vtool_name
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        return path

    @staticmethod
    def _write_csv(directory: Path, filename: str, df: pd.DataFrame) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        df.to_csv(path, sep=";", index=False)
        return path
