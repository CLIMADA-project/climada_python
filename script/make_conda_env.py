#!/usr/bin/env python
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML"]
# ///
import tomllib
from pathlib import Path
from typing import Any

import yaml

CONVERSIION_TABLE = {"matplotlib": "matplotlib-base"}

ADDITIONS = ["pip"]
DELETIONS = []


def read_pyproject(path: str | Path = "pyproject.toml") -> dict[str, Any]:
    """Read the pyproject file as dict"""
    with open(path, "rb") as file:
        return tomllib.load(file)


def read_all_dependencies(pyproj_data: dict[str, Any]) -> list[str]:
    """Read all dependencies (also optional) and merge them"""
    dependencies: list[str] = pyproj_data["project"]["dependencies"]
    if "optional-dependencies" in pyproj_data["project"]:
        for dep_list in pyproj_data["project"]["optional-dependencies"].values():
            dependencies.extend(dep_list)
    return dependencies


def remove_dependencies(dep_list: list[str]) -> list[str]:
    """Remove dependencies"""

    def delete(dependency: str) -> bool:
        for dep in DELETIONS:
            if dep in dependency:
                return True
        return False

    return [dep for dep in dep_list if not delete(dep)]


def convert_dependencies(dep_list: list[str]) -> list[str]:
    """Convert dependency names"""

    def convert(dependency: str) -> str:
        for dep_from, dep_to in CONVERSIION_TABLE.items():
            if dep_from in dependency:
                return dependency.replace(dep_from, dep_to)
        return dependency

    return [convert(dep) for dep in dep_list]


def add_dependencies(dep_list: list[str]) -> list[str]:
    """Add dependencies"""
    dep_list = dep_list.copy()
    dep_list.extend(ADDITIONS)
    return dep_list


def write_environment_file(
    dependencies: list[str],
    output_path: str | Path = "./requirements/env_climada.yml",
):
    """Write the conda environment file"""
    data = {
        "name": "climada_env",
        "channels": ["conda-forge", "nodefaults"],
        "dependencies": sorted(dependencies),
    }
    with open(output_path, "w") as file:
        yaml.dump(data, file)


def main():
    pyproject_data = read_pyproject()
    dependencies = read_all_dependencies(pyproject_data)
    dependencies = remove_dependencies(dependencies)
    dependencies = convert_dependencies(dependencies)
    dependencies = add_dependencies(dependencies)
    write_environment_file(dependencies)


if __name__ == "__main__":
    main()
