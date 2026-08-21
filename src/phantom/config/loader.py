"""
Provides functionality to locate and load the project's TOML
configuration file.
"""

from pathlib import Path
import sys
import tomllib


class ConfigLoader:
    # Root of this codebase (where config.toml itself lives).
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Root for the data workspace that config.toml's paths describe.
    # That workspace is a sibling of the project, not part of it: the
    # pipeline reads/writes genomes, features, results, etc. next to the
    # codebase, never inside it. Resolve every path taken from config.toml
    # through `resolve_data_path()` below rather than joining it onto
    # PROJECT_ROOT directly.
    DATA_ROOT = PROJECT_ROOT.parent

    DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

    def __init__(self, path: Path | None = None):
        self.path = path or self.DEFAULT_CONFIG_PATH
        self.config = None

    @classmethod
    def resolve_data_path(cls, relative_path: str | Path) -> Path:
        """
        Resolves a path declared in config.toml (e.g. "data/genomes")
        against DATA_ROOT, the data workspace sitting next to the project.
        """
        return cls.DATA_ROOT / relative_path

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
    