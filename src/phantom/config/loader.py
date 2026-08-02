"""
@author: Radosław Pławecki
"""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config(path: Path = CONFIG_PATH):
    with open(path, "rb") as f:
        return tomllib.load(f)
    