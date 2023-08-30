#!/usr/bin/env python3
"""This script is part of the GitHub CI make-release pipeline

It reads the version number from climada*/_version.py and then uses the `gh` cli
to create the new release.

"""
import glob
import re
import subprocess


def get_version() -> str:
    """Return the current version number, based on the _version.py file."""
    [version_file] = glob.glob("climada*/_version.py")
    with open(version_file, 'r', encoding="UTF-8") as vfp:
        content = vfp.read()
    regex = r'^__version__\s*=\s*[\'\"](.*)[\'\"]\s*$'
    mtch = re.match(regex, content)
    return mtch.group(1)


def make_release():
    """run `gh release create vX.Y.Z"""
    version_number = get_version()
    subprocess.run(
        ["gh", "release", "create", "--generate-notes", f"v{version_number}"],
        check=True,
    )


if __name__ == "__main__":
    make_release()
