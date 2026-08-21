"""
Safely reads and updates config.toml.
"""

from pathlib import Path
import tomlkit
from phantom.config.loader import ConfigLoader


class FeatureConfigManager:
    def __init__(self, config_path: Path | str | None = None):
        self.config_path = Path(config_path) if config_path else ConfigLoader.DEFAULT_CONFIG_PATH

    def load_doc(self) -> tomlkit.TOMLDocument:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return tomlkit.parse(f.read())

    def get_existing_versions(self) -> list[str]:
        doc = self.load_doc()
        if "features" not in doc:
            return []
        versions = []
        for _, value in doc["features"].items():
            if isinstance(value, dict) and "version" in value:
                versions.append(value["version"])
        return versions

    def create_new_version(self, version_str: str) -> Path:
        doc = self.load_doc()
        if "features" not in doc:
            doc.add("features", tomlkit.table())
        v_key = f"v{version_str.replace('.', '_')}"
        if v_key in doc["features"]:
            raise ValueError(f"Version {version_str} already exists in config.")
        features_table = doc["features"]
        features_root = features_table.get("path", "data/features")
        features_table.add(tomlkit.nl())
        features_table.add(tomlkit.nl())
        v_table = tomlkit.table()
        base_path = f"{features_root}/{version_str}"
        v_table.add("path", base_path)
        v_table.add("version", version_str)
        v_table.add("type", "directory")
        sub_dirs = {
            "raw": ("raw", "Raw feature files stored separately for each sample before merging and preprocessing."),
            "raw_merged": ("raw-merged", "Merged raw feature data combined into unified files containing all samples."),
            "preprocessed": ("preprocessed", "Preprocessed feature data."),
            "extracted": ("extracted", "Extracted feature representations used as input for classifier training and prediction.")
        }
        for sub_key, (folder_name, desc) in sub_dirs.items():
            sub_table = tomlkit.table()
            sub_table.add("path", folder_name)
            sub_table.add("type", "directory")
            sub_table.add("description", desc)
            v_table.add(sub_key, sub_table)
        features_table.add(v_key, v_table)
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))
        return ConfigLoader.resolve_data_path(base_path) / "raw"

    def get_version_path(self, version_str: str) -> Path:
        return ConfigLoader.resolve_data_path(self._get_version_table(version_str)["path"])

    def get_stage_dir(self, version_str: str, stage: str) -> Path:
        """
        Resolves a version's stage directory (e.g. "raw", "raw_merged",
        "preprocessed", "extracted") from its config.toml entry, rather
        than assuming a hardcoded folder name -- config.toml is the single
        source of truth for how each stage's directory is named.
        """
        version_table = self._get_version_table(version_str)
        if stage not in version_table:
            raise ValueError(
                f"Stage '{stage}' not found for version {version_str} in config."
            )
        return self.get_version_path(version_str) / version_table[stage]["path"]

    def _get_version_table(self, version_str: str) -> dict:
        doc = self.load_doc()
        v_key = f"v{version_str.replace('.', '_')}"
        features_table = doc.get("features", {})
        if v_key not in features_table:
            raise ValueError(f"Version {version_str} not found in config.")
        return features_table[v_key]
