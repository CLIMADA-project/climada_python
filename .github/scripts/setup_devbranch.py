#!/usr/bin/env python3
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
    with open("CHANGELOG.md", 'r') as cl:
        lines = cl.readlines()

    if "## Unreleased" in lines:
        return

    with open("CHANGELOG.md", 'w') as cl:
        cl.write("""# Changelog

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

""")
        cl.writelines(lines[2:])


def update_version(nvn):
    """Update the _version.py file"""
    [file_with_version] = glob.glob("climada*/_version.py")
    regex = r'(^__version__\s*=\s*[\'\"]).*([\'\"]\s*$)'
    return update_file(file_with_version, regex, nvn)


def update_setup(new_version_number):
    """Update the setup.py file"""
    file_with_version = "setup.py"
    regex = r'(^\s+version\s*=\s*[\'\"]).*([\'\"]\s*,\s*$)'
    return update_file(file_with_version, regex, new_version_number)


def update_file(file_with_version, regex, new_version_number):
    """Replace the version number(s) in a file, based on a rgular expression."""
    with open(file_with_version, 'r') as curf:
        lines = curf.readlines()
    successfully_updated = False
    for i, line in enumerate(lines):
        m = re.match(regex, line)
        if m:
            lines[i] = f"{m.group(1)}{new_version_number}{m.group(2)}"
            successfully_updated = True
    if not successfully_updated:
        raise Exception(f"cannot determine version of {file_with_version}")
    with open(file_with_version, 'w') as newf:
        for line in lines:
            newf.write(line)


def setup_devbranch():
    """Adjust files after a release was published, i.e.,
    apply the canonical deviations from main in develop.
    
    Just changes files, all `git` commands are in the setup_devbranch.sh file.
    """
    main_version = get_last_version().strip('v')
    
    dev_version = f"{main_version}-dev"

    update_setup(dev_version)
    update_version(dev_version)
    update_changelog()

    print(f"v{dev_version}")


if __name__ == "__main__":
    setup_devbranch()

