"""
Contains CLI prompts used during the feature processing workflow.
"""

from pathlib import Path
import questionary
import pandas as pd
from phantom.config.loader import ConfigLoader


class FeatureCollectionPrompts:
    @staticmethod
    def ask_action() -> str:
        return questionary.select(
            "Select feature processing action:",
            choices=[
                "1) Import external features",
                "2) Merge raw feature files",
                "3) Preprocess features",
                "4) Extract features",
                "5) Optimize features",
                "Back",
            ],
        ).ask()

    @staticmethod
    def ask_auto_merge(version: str) -> bool:
        """Prompts the user to automatically merge raw feature files after importing."""
        return questionary.confirm(
            f"Would you like to automatically merge raw feature files for version {version} now?"
        ).ask()

    @staticmethod
    def ask_feature_paths() -> list[Path]:
        print("\n[INFO] Select the external feature directories/files to import.")
        print("       (Leave the input blank and press Enter when you are done adding paths)")
        paths = []
        while True:
            p_str = questionary.path("Path to feature data:").ask()
            if not p_str:
                break
            p = Path(p_str)
            if not p.exists():
                print(f"[ERROR] Path does not exist: {p}")
                continue
            paths.append(p)
        return paths

    @staticmethod
    def ask_version_strategy(existing_versions: list[str]) -> str:
        choices = ["Create a new feature version"]
        if existing_versions:
            choices.append("Append to an existing version")
        choices.append("Cancel")
        return questionary.select(
            "How would you like to version these features?",
            choices=choices
        ).ask()

    @staticmethod
    def ask_new_version() -> str:
        return questionary.text(
            "Enter the new version number (e.g., '1.0', '1.1', '2.0'):",
            validate=lambda text: len(text) > 0 and "." in text 
            or "Version should typically be in 'X.Y' format."
        ).ask()

    @staticmethod
    def ask_existing_version(versions: list[str]) -> str:
        return questionary.select("Select target version:", choices=versions).ask()

    @staticmethod
    def ask_id_column(tool_path_str: str, columns: list[str]) -> str:
        return questionary.select(
            f"Select the sequence/contig ID column for [{tool_path_str}]:",
            choices=columns
        ).ask()


class FeatureExtractionPrompts:
    def __init__(self, config: dict | None = None):
        loader = ConfigLoader()
        self.config = config if config is not None else loader.load()
        self.tool_map = self.config.get("tools", {})
        mask_path_str = self.config.get("masks", {}).get("path")
        self.mask_root = (
            ConfigLoader.resolve_data_path(mask_path_str) if mask_path_str else None
        )

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


class FeatureOptimizationPrompts:
    @staticmethod
    def ask_tool_choice(tools: list[str]) -> str | None:
        return questionary.select("Select tool to optimize:", choices=tools).ask()

    @staticmethod
    def ask_file_choice(label: str, files: list[Path]) -> Path | None:
        if not files:
            return None
        if len(files) == 1:
            return files[0]
        choice = questionary.select(
            f"Select {label} file:",
            choices=[f.name for f in files]
        ).ask()
        if not choice:
            return None
        return next(f for f in files if f.name == choice)
