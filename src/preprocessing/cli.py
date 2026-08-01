"""
@author: Radosław Pławecki
"""

import questionary
import pandas as pd
from pathlib import Path
from config.loader import load_config

tool_map = {
    "geN": "genomad",
    "VIB": "vibrant",
    "VS2": "virsorter2",
}

mask_root = Path("data/masks")


def ask_column(df: pd.DataFrame) -> str:
    selectable_columns = [col for col in df.columns if col not in {'Accession', 'id'}]
    if not selectable_columns:
        raise ValueError("No valid columns to choose from.")
    col = questionary.select(
        "Choose a column:",
        choices=selectable_columns
    ).ask()
    print(f"\n[INFO] Selected column: {col}")
    return col


def ask_mask_file(in_root: Path) -> Path | None:
    tool_id = in_root.stem.split("_")[0]         
    tool = tool_map.get(tool_id)
    if tool is None:
        raise ValueError(f"Unknown tool identifier: {tool_id}")
    mask_dir = mask_root / tool
    masks = sorted(mask_dir.glob("*.tsv"))
    if not masks:
        print(f"[INFO] No masks found in {mask_dir}")
        return None
    choice = questionary.select(
        "Select CheckV mask:",
        choices=[m.name for m in masks] + ["None"],
    ).ask()
    if choice == "None":
        return None
    return mask_dir / choice


def ask_feature_method() -> str:
    method = questionary.select(
        "Select feature engineering method for the phage-host relation:",
        choices=[
            "1) Predation Pressure [pp]",
            "2) Occurrence Matrix [om]",
        ],
    ).ask()
    print(f"\n[INFO] Selected feature engineering method: {method}")
    return method


def ask_normalization_method() -> str:
    method = questionary.select(
        "Select normalization method:",
        choices=[
            "1) TSS + Z-score [tss_z]",
            "2) CLR + Z-score [clr_z]",
            "3) Only TSS [tss]",
            "4) Nothing (raw data) [raw]",
        ],
    ).ask()
    print(f"[INFO] Selected normalization: {method}")
    return method
    

from pathlib import Path


class ModalityFileSelector:
    def __init__(self, modality: str, config_path="project/config.toml"):
        self.modality = modality
        self.config = load_config(config_path)
        self.modalities_config = self.config["modalities"]

    def select(self) -> Path:
        version = self.select_version()
        raw_merged = self._get_raw_merged_path(version)
        modality_dir = raw_merged / self.modality
        return self.select_file(modality_dir)

    def select_version(self):
        versions = [key for key in self.config["modalities"] if key.startswith("v")]
        return questionary.select(
            "Select modality version:",
            choices=versions
        ).ask()

    def _get_raw_merged_path(self, version):
        version_config = (self.modalities_config[version])
        root = Path(version_config["path"])
        raw_merged = (root / version_config["raw_merged"]["path"])
        return raw_merged

    def select_file(self, directory: Path):
        files = sorted(directory.glob("*.csv"))
        if len(files) == 1:
            return files[0]
        selected = questionary.select(
            f"Select {self.modality} file:",
            choices=[
                file.name
                for file in files
            ]
        ).ask()
        return directory / selected
