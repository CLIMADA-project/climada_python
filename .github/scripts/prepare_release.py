#!/usr/bin/env python3
"""This script is part of the GitHub CI make-release pipeline

The following preparation steps are executed:

- update version numbers in _version.py and setup.py
- purge the "Unreleased" section of CHANGELOG.md and rename it to the new version number

All changes are immediately commited to the repository.
"""

import glob
import json
import re
import subprocess
import time


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


def bump_version_number(version_number: str, level: str) -> str:
    """Return a copy of `version_number` with one level number incremented."""
    major, minor, patch = version_number.split(".")
    if level == "major":
        major = str(int(major) + 1)
        minor = "0"
        patch = "0"
    elif level == "minor":
        minor = str(int(minor) + 1)
        patch = "0"
    elif level == "patch":
        patch = str(int(patch) + 1)
    else:
        raise ValueError(f"level should be 'major', 'minor' or 'patch', not {level}")
    return ".".join([major, minor, patch])


def update_changelog(nvn):
    """Rename the "Unreleased" section, remove unused subsections and the code-freeze date,
    set the release date to today"""
    releases = []
    release_name = None
    release = []
    section_name = None
    section = []
    with open("CHANGELOG.md", "r", encoding="UTF-8") as changelog:
        for line in changelog.readlines():
            if line.startswith("#"):
                if line.startswith("### "):
                    if section:
                        release.append((section_name, section))
                    section_name = line[4:].strip()
                    section = []
                    # print("tag:", section_name)
                elif line.startswith("## "):
                    if section:
                        release.append((section_name, section))
                    if release:
                        releases.append((release_name, release))
                    release_name = line[3:].strip()
                    release = []
                    section_name = None
                    section = []
                    # print("release:", release_name)
            else:
                section.append(line)
        if section:
            release.append((section_name, section))
        if release:
            releases.append((release_name, release))

    with open("CHANGELOG.md", "w", encoding="UTF-8") as changelog:
        changelog.write("# Changelog\n\n")
        for release_name, release in releases:
            if release_name:
                if release_name.lower() == "unreleased":
                    release_name = nvn
                changelog.write(f"## {release_name}\n")
            for section_name, section in release:
                if any(ln.strip() for ln in section):
                    if section_name:
                        changelog.write(f"### {section_name}\n")
                    lines = [
                        ln.strip()
                        for ln in section
                        if "code freeze date: " not in ln.lower()
                    ]
                    if not section_name and release_name.lower() == nvn:
                        print("setting date")
                        for i, line in enumerate(lines):
                            if "release date: " in line.lower():
                                today = time.strftime("%Y-%m-%d")
                                lines[i] = f"Release date: {today}"
                    changelog.write(re.sub("\n+$", "\n", "\n".join(lines)))
                    changelog.write("\n")
    return GitFile("CHANGELOG.md")


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
    return GitFile(file_with_version)


class GitFile:
    """Helper class for `git add`."""

    def __init__(self, path):
        self.path = path

    def gitadd(self):
        """run `git add`"""
        _gitadd = subprocess.run(
            ["git", "add", self.path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf8")


class Git:
    """Helper class for `git commit`."""

    def __init__(self):
        _gitname = subprocess.run(
            ["git", "config", "--global", "user.name", "'climada'"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf8")
        _gitemail = subprocess.run(
            ["git", "config", "--global", "user.email", "'test.climada@gmail.com'"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf8")

    def commit(self, new_version):
        """run `git commit`."""
        try:
            _gitcommit = subprocess.run(
                ["git", "commit", "-m", f"'Automated update v{new_version}'"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf8")
            _gitpush = subprocess.run(
                ["git", "push"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf8")
        except subprocess.CalledProcessError as err:
            message = err.stdout.decode("utf8")
            print("message:", message)
            if "nothing to commit" in message:
                print("repo already up to date with new version number")
            else:
                raise RuntimeError(f"failed to run: {message}") from err


def prepare_new_release(level):
    """Prepare files for a new release on GitHub."""
    try:
        last_version_number = get_last_version().strip("v")
    except subprocess.CalledProcessError as err:
        if "release not found" in err.stderr.decode("utf8"):
            # The project doesn't have any releases yet.
            last_version_number = "0.0.0"
        else:
            raise
    new_version_number = bump_version_number(last_version_number, level)

    update_setup(new_version_number).gitadd()
    update_version(new_version_number).gitadd()
    update_changelog(new_version_number).gitadd()

    Git().commit(new_version_number)


if __name__ == "__main__":
    from sys import argv

    try:
        LEVEL = argv[1]
    except IndexError:
        LEVEL = "patch"
    prepare_new_release(LEVEL)
