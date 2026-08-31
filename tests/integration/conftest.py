"""Auto-mark every test collected from tests/integration/ as ``integration``."""

from pathlib import Path

import pytest

_HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if _HERE in Path(item.path).parents:
            item.add_marker(pytest.mark.integration)
