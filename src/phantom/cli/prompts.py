"""
@author: Radosław Pławecki
"""

import questionary
import pandas as pd
from pathlib import Path
from phantom.config.loader import load_config


class FeatureExtractionPrompts:
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else load_config()
        self.tool_map = self.config.get("tools", {})
        mask_path_str = self.config.get("masks", {}).get("path")
        self.mask_root = Path(mask_path_str) if mask_path_str else None

    def _get_tool_mask_dir(self, in_root: Path) -> Path | None:
        if not self.mask_root:
            print("[WARNING] [masks][path] not found in config.toml.")
            return None
        tool_prefix = in_root.stem.split("_")[0]
        tool_name = self.tool_map.get(tool_prefix)
        if tool_name is None:
            print(f"[WARNING] Unknown tool prefix '{tool_prefix}'. Please add it to the [tools] section in config.toml.")
            return None
        return self.mask_root / tool_name

    def _get_available_masks(self, mask_dir: Path) -> list[Path]:
        if not mask_dir.exists():
            print(f"[INFO] Mask directory does not exist: {mask_dir}.")
            return []
        masks = sorted(mask_dir.glob("*.tsv"))
        if not masks:
            print(f"[INFO] No masks (*.tsv) found in {mask_dir}.")
            return []
        return masks
    
    def ask_mask_file(self, in_root: Path) -> Path | None:
        wants_mask = questionary.confirm("Do you want to apply a CheckV mask?").ask()
        if not wants_mask:
            return None
        mask_dir = self._get_tool_mask_dir(in_root)
        if not mask_dir:
            return None
        masks = self._get_available_masks(mask_dir)
        if not masks:
            return None
        choice = questionary.select(
            f"Select CheckV mask for {mask_dir.name}:",
            choices=[m.name for m in masks] + ["None"],
        ).ask()
        if choice == "None":
            return None
        selected_path = mask_dir / choice
        print(f"\n[INFO] Selected mask: {selected_path}")
        return selected_path

    def ask_column(self, df: pd.DataFrame) -> str:
        selectable_columns = [col for col in df.columns if col not in {'Accession', 'id'}]
        if not selectable_columns:
            raise ValueError("No valid columns to choose from.")
        col = questionary.select(
            "Choose a column:",
            choices=selectable_columns
        ).ask()
        print(f"\n[INFO] Selected column: {col}")
        return col

    def ask_feature_method(self) -> str:
        method = questionary.select(
            "Select feature engineering method for the phage-host relation:",
            choices=[
                "1) Predation Pressure [pp]",
                "2) Occurrence Matrix [om]",
            ],
        ).ask()
        print(f"\n[INFO] Selected feature engineering method: {method}")
        return method

    def ask_normalization_method(self) -> str:
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

    def ask_min_patients(self) -> int:
        answer = questionary.text(
            "Specify the minimum number of patients (from 1 to 10):",
            validate=lambda text: (
                text.isdigit() and 1 <= int(text) <= 10
            ) or "Please enter a number between 1 and 10."
        ).ask()
        return int(answer)

    def ask_binary(self) -> bool:
        return questionary.confirm(
            "Use binary representation?"
        ).ask()

    def ask_min_pident(self) -> float:
        answer = questionary.text(
            "Specify the minimum sequence identity (e.g. 0.35):",
            validate=self._validate_float
        ).ask()
        return float(answer)

    def ask_min_coverage(self) -> float:
        answer = questionary.text(
            "Specify the minimum coverage (e.g. 0.70):",
            validate=self._validate_float
        ).ask()
        return float(answer)

    @staticmethod
    def _validate_float(text: str) -> bool | str:
        try:
            val = float(text)
            if 0.0 <= val <= 1.0:
                return True
            return "Value must be between 0.0 and 1.0."
        except ValueError:
            return "Please enter a valid number (e.g. 0.35)."


class ModalityFileSelector:
    def __init__(self, modality: str, config_path="project/config.toml"):
        self.modality = modality
        self.config = load_config(config_path)
        self.modalities_config = self.config["modalities"]
        self.selected_version = None

    def select(self) -> Path:
        self.selected_version = self.select_version()
        raw_merged = self.get_stage_dir("raw_merged")
        return self.select_file(raw_merged)

    def select_version(self):
        versions = [key for key in self.config["modalities"] if key.startswith("v")]
        return questionary.select(
            "Select modality version:",
            choices=versions
        ).ask()

    def get_stage_dir(self, stage: str) -> Path:
        """
        Dynamically fetches any directory path from config.toml 
        (e.g., 'raw_merged', 'preprocessed', 'features')
        """
        if not self.selected_version:
            raise ValueError("You must call select() or select_version() first.")
        version_config = self.modalities_config[self.selected_version]
        root = Path(version_config["path"])
        stage_path = Path(version_config[stage]["path"])
        return root / stage_path / self.modality

    def select_file(self, directory: Path):
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {directory}")
        if len(files) == 1:
            return files[0]
        selected = questionary.select(
            f"Select {self.modality} file:",
            choices=[file.name for file in files]
        ).ask()
        return directory / selected
