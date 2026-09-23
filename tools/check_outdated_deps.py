"""
Filters `pip list --outdated` down to the packages this project actually
declares in pyproject.toml (dependencies + dev extras) -- the dev
environment here is a shared conda-like env with hundreds of unrelated
packages, so the unfiltered output is mostly noise.
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def project_package_names() -> set[str]:
    project = tomllib.loads(PYPROJECT_PATH.read_text())["project"]
    specs = list(project.get("dependencies", []))
    for extra_deps in project.get("optional-dependencies", {}).values():
        specs.extend(extra_deps)
    names = set()
    for spec in specs:
        match = re.match(r"^[A-Za-z0-9._-]+", spec)
        if match:
            names.add(normalize(match.group(0)))
    return names


def main() -> int:
    wanted = project_package_names()
    result = subprocess.run(
        ["pip", "list", "--outdated", "--format=json"],
        capture_output=True,
        text=True,
        check=True,
    )
    outdated = [pkg for pkg in json.loads(result.stdout) if normalize(pkg["name"]) in wanted]
    if not outdated:
        print("All project dependencies (per pyproject.toml) are up to date.")
        return 0
    name_width = max(len(pkg["name"]) for pkg in outdated)
    version_width = max(len(pkg["version"]) for pkg in outdated)
    for pkg in outdated:
        print(
            f"{pkg['name']:<{name_width}}  {pkg['version']:<{version_width}} -> {pkg['latest_version']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
