"""
Controls the feature pipeline workflow: import/collection, merging, 
preprocessing and extraction.
"""

from pathlib import Path

from phantom.config.loader import ConfigLoader
from phantom.config.features import FeatureConfigManager
from phantom.cli.features import FeatureCollectionPrompts
from phantom.features.collection.importer import FeatureImporter
from phantom.features.merging.concatenator import FeatureConcatenator
from phantom.features.preprocessing.preprocessor import FeaturePreprocessor
from phantom.features.extraction.extractor import FeatureExtractor
from phantom.features.optimization.optimizer import FeatureOptimizer


class FeatureController:
    def __init__(self, config_path: Path | str | None = None):
        loader = ConfigLoader(Path(config_path) if config_path else None)
        self.config = loader.load()
        self.config_path = loader.path
        self.config_mgr = FeatureConfigManager(self.config_path)

    def run(self):
        while True:
            print("\n--- Feature Processing ---")
            action = FeatureCollectionPrompts.ask_action()
            if action is None or action == "Back":
                break
            if action.startswith("1)"):
                self._import_features()
            elif action.startswith("2)"):
                self._merge_features()
            elif action.startswith("3)"):
                self._preprocess_features()
            elif action.startswith("4)"):
                self._extract_features()
            elif action.startswith("5)"):
                self._optimize_features()

    def _import_features(self) -> None:
        importer = FeatureImporter(config_path=self.config_path)
        raw_path = importer.run()
        if raw_path and raw_path.exists():
            version = raw_path.parent.name
            if FeatureCollectionPrompts.ask_auto_merge(version):
                concatenator = FeatureConcatenator(
                    version=version,
                    config_mgr=self.config_mgr
                )
                concatenator.run()

    def _merge_features(self) -> None:
        existing_versions = self.config_mgr.get_existing_versions()
        if not existing_versions:
            print("[ERROR] No feature versions found in config.toml. Import features first.")
            return
        version = FeatureCollectionPrompts.ask_existing_version(existing_versions)
        if version:
            concatenator = FeatureConcatenator(
                version=version,
                config_mgr=self.config_mgr
            )
            concatenator.run()

    def _preprocess_features(self):
        existing_versions = self.config_mgr.get_existing_versions()
        if not existing_versions:
            print("[ERROR] No feature versions available to preprocess.")
            return
        version = FeatureCollectionPrompts.ask_existing_version(existing_versions)
        if version:
            FeaturePreprocessor(version=version, config_mgr=self.config_mgr).run()

    def _extract_features(self):
        existing_versions = self.config_mgr.get_existing_versions()
        if not existing_versions:
            print("[ERROR] No feature versions available for extraction.")
            return
        version = FeatureCollectionPrompts.ask_existing_version(existing_versions)
        if version:
            FeatureExtractor(version=version, config_mgr=self.config_mgr).run()

    def _optimize_features(self):
        FeatureOptimizer(config_mgr=self.config_mgr).run()

if __name__ == "__main__":
    FeatureController().run()
