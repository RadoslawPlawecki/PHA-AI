"""
@author: Radosław Pławecki
"""

import questionary
import pandas as pd
from pathlib import Path

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
    