"""
Coordinates importing feature sources into the project structure.
"""

import shutil
from pathlib import Path
import questionary
from phantom.config.features import FeatureConfigManager
from phantom.cli.features import FeatureCollectionPrompts


class FeatureImporter:
    def __init__(self, config_path: Path | str | None = None):
        self.config_mgr = FeatureConfigManager(config_path)

    def run(self) -> Path | None:
        print("\n--- Feature Import & Versioning ---")
        source_paths = FeatureCollectionPrompts.ask_feature_paths()
        if not source_paths:
            print("[INFO] No paths provided. Aborting.")
            return None
        existing_versions = self.config_mgr.get_existing_versions()
        strategy = FeatureCollectionPrompts.ask_version_strategy(existing_versions)
        if strategy == "Cancel" or not strategy:
            return None
        target_raw_path = None
        if strategy == "Create a new feature version":
            version_str = FeatureCollectionPrompts.ask_new_version()
            if not version_str:
                return None
            if version_str in existing_versions:
                print(f"[ERROR] Version {version_str} already exists. Aborting.")
                return None
            print(f"\n[INFO] Updating config.toml with v{version_str} schema...")
            target_raw_path = self.config_mgr.create_new_version(version_str)
        elif strategy == "Append to an existing version":
            version_str = FeatureCollectionPrompts.ask_existing_version(existing_versions)
            if not version_str:
                return None
            target_raw_path = self.config_mgr.get_version_path(version_str) / "raw"
        if target_raw_path:
            self._copy_features(source_paths, target_raw_path)
            return target_raw_path
        return None

    def _copy_features(self, sources: list[Path], dest_dir: Path):
        print(f"\n[INFO] Copying data to {dest_dir} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        for src in sources:
            target_path = dest_dir / src.name
            if target_path.exists():
                overwrite = questionary.confirm(
                    f"[WARNING] {src.name} already exists in {dest_dir.name}. Overwrite?"
                ).ask()
                if not overwrite:
                    print(f"     Skipped: {src.name}")
                    continue
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            try:
                if src.is_dir():
                    shutil.copytree(src, target_path)
                else:
                    shutil.copy2(src, target_path)
                success_count += 1
                print(f"     Copied: {src.name}")
            except Exception as e:
                print(f"[ERROR] Error copying {src.name}: {e}")
        print(f"\n[SUCCESS] Imported {success_count} source(s) into {dest_dir}.")
