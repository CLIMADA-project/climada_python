#!/usr/bin/env python3
"""This script is part of the GitHub CI postrelease-setup-devbranch pipeline

The following preparation steps are executed:

- update version numbers in _version.py and setup.py: append a -dev suffix
- insert a vanilla "unreleased" section on top of CHANGELOG.md

The changes are not commited to the repository. This is dealt with in the bash script
`setup_devbranch.sh` (which is also the caller of this script).
"""
import glob
import json
import re
import subprocess


def get_last_version() -> str:
    """Return the version number of the last release."""
    json_string = (
        subprocess.run(
            ["gh", "release", "view", "--json", "tagName"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf8")
        .strip()
    )

    return json.loads(json_string)["tagName"]


def update_changelog():
    """Insert a vanilla "Unreleased" section on top."""
    with open("CHANGELOG.md", "r", encoding="UTF-8") as changelog:
        lines = changelog.readlines()

    if "## Unreleased" in lines:
        return

    with open("CHANGELOG.md", "w", encoding="UTF-8") as changelog:
        changelog.write(
            """# Changelog

## Unreleased

Release date: YYYY-MM-DD

Code freeze date: YYYY-MM-DD

### Description

### Dependency Changes

### Added

### Changed

### Fixed

### Deprecated

### Removed

"""
        )
        changelog.writelines(lines[2:])


def update_version(nvn):
    """Update the _version.py file"""
    [file_with_version] = glob.glob("climada*/_version.py")
    regex = r"(^__version__\s*=\s*[\'\"]).*([\'\"]\s*$)"
    return update_file(file_with_version, regex, nvn)


def update_setup(new_version_number):
    """Update the setup.py file"""
    file_with_version = "setup.py"
    regex = r"(^\s+version\s*=\s*[\'\"]).*([\'\"]\s*,\s*$)"
    return update_file(file_with_version, regex, new_version_number)


def update_file(file_with_version, regex, new_version_number):
    """Replace the version number(s) in a file, based on a rgular expression."""
    with open(file_with_version, "r", encoding="UTF-8") as curf:
        lines = curf.readlines()
    successfully_updated = False
    for i, line in enumerate(lines):
        mtch = re.match(regex, line)
        if mtch:
            lines[i] = f"{mtch.group(1)}{new_version_number}{mtch.group(2)}"
            successfully_updated = True
    if not successfully_updated:
        raise RuntimeError(f"cannot determine version of {file_with_version}")
    with open(file_with_version, "w", encoding="UTF-8") as newf:
        for line in lines:
            newf.write(line)


def setup_devbranch():
    """Adjust files after a release was published, i.e.,
    apply the canonical deviations from main in develop.

    Just changes files, all `git` commands are in the setup_devbranch.sh file.
    """
    main_version = get_last_version().strip("v")
    semver = main_version.split(".")
    semver[-1] = f"{int(semver[-1]) + 1}-dev"
    dev_version = ".".join(semver)

    update_setup(dev_version)
    update_version(dev_version)
    update_changelog()

    print(f"v{dev_version}")


if __name__ == "__main__":
    setup_devbranch()
