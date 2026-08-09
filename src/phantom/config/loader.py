"""
Provides functionality to locate and load the project's TOML
configuration file.
"""

from pathlib import Path
import sys
import tomllib


class ConfigLoader:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

    def __init__(self, path: Path | None = None):
        self.path = path or self.DEFAULT_CONFIG_PATH
        self.config = None
        
    def load(self) -> dict:
        if not self.path.is_file():
            print(f"\n[ERROR] Configuration file '{self.path}' not found.")
            print(
                "Please ensure you have prepared the configuration file "
                "in advance according to the SOP."
            )
            sys.exit(1)
        with open(self.path, "rb") as f:
            self.config = tomllib.load(f)
        return self.config
    