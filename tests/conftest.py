"""Session fixtures and test isolation for the phantom suite.

See the testing strategy for rationale. Key jobs here:
- force a headless matplotlib backend (Visualizer writes PDFs);
- silence Optuna;
- tear down phantom's module-global logger handlers so FileHandlers to old
  tmp dirs do not leak across tests;
- give a deterministic legacy numpy seed (pytest-randomly still shuffles order);
- redirect ConfigLoader's data workspace at a tmp tree (stop-gap until the
  root becomes an env/instance param -- refactor R4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _mpl_agg():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture(scope="session", autouse=True)
def _quiet_optuna():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    for name in ("tool", "phantom"):
        lg = logging.getLogger(name)
        for handler in list(lg.handlers):
            handler.close()
            lg.removeHandler(handler)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(1234)


@pytest.fixture
def phantom_data_root(tmp_path, monkeypatch):
    """Point ConfigLoader's class-level roots at a throwaway tree so nothing
    resolves against the real ``../data`` workspace or the repo config.toml.
    """
    from phantom.config.loader import ConfigLoader

    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    monkeypatch.setattr(ConfigLoader, "DATA_ROOT", root)
    monkeypatch.setattr(ConfigLoader, "PROJECT_ROOT", root)
    monkeypatch.setattr(ConfigLoader, "DEFAULT_CONFIG_PATH", root / "config.toml")
    return root


@pytest.fixture
def minimal_config_text() -> str:
    return (FIXTURES / "config" / "config.minimal.toml").read_text(encoding="utf-8")


@pytest.fixture
def tmp_config(tmp_path, minimal_config_text) -> Path:
    """A writable copy of the minimal config.toml (for FeatureConfigManager,
    which mutates the file in place)."""
    path = tmp_path / "config.toml"
    path.write_text(minimal_config_text, encoding="utf-8")
    return path


@pytest.fixture
def tmp_workspace(phantom_data_root, minimal_config_text):
    """A full throwaway data workspace: config.toml + the feature-version
    stage tree + mask directories, with ConfigLoader pointed at it."""
    from tests._helpers.workspace import STAGE_DIRS, VTOOLS, Workspace

    config_path = phantom_data_root / "config.toml"
    config_path.write_text(minimal_config_text, encoding="utf-8")
    ws = Workspace(root=phantom_data_root, config_path=config_path)
    for stage in STAGE_DIRS:
        ws.stage_dir(stage).mkdir(parents=True, exist_ok=True)
    for vtool_name in VTOOLS.values():
        (ws.data_root / "masks" / vtool_name).mkdir(parents=True, exist_ok=True)
    return ws
