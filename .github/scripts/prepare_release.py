#!/usr/bin/env python3
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
        major = str(int(major)+1)
    elif level == "minor":
        minor = str(int(minor)+1)
    elif level == "patch":
        patch = str(int(patch)+1)
    else:
        raise ValueError(f"level should be 'major', 'minor' or 'patch', not {level}")
    return ".".join([major, minor, patch])


def update_readme(nvn):
    """align doc/misc/README.md with ./README.md but remove the non-markdown header lines from """
    with open("README.md", 'r') as rmin:
        lines = [line for line in rmin.readlines() if not line.startswith('[![')]
    while not lines[0].strip():
        lines = lines[1:]
    with open("doc/misc/README.md", 'w') as rmout:
        rmout.writelines(lines)
    return GitFile('doc/misc/README.md')


def update_changelog(nvn):
    """Rename the "Unreleased" section, remove unused subsections and the code-freeze date,
    set the release date to today"""
    r = [] 
    crn = None
    cr = [] 
    ctn = None
    ct = []
    with open("CHANGELOG.md", 'r') as cl:
        for line in cl.readlines():
            if line.startswith('#'):
                if line.startswith('### '):
                    if ct:
                        cr.append((ctn, ct))
                    ctn = line[4:].strip()
                    ct = []
                    #print("tag:", ctn)
                elif line.startswith('## '):
                    if ct:
                        cr.append((ctn, ct))
                    if cr:
                        r.append((crn, cr))
                    crn = line[3:].strip()
                    cr = []
                    ctn = None
                    ct = []
                    #print("release:", crn)
            else:
                ct.append(line)
        if ct:
            cr.append((ctn, ct))
        if cr:
            r.append((crn, cr))

    with open("CHANGELOG.md", 'w') as cl:
        cl.write("# Changelog\n\n")
        for crn, cr in r:
            if crn:
                if crn.lower() == "unreleased":
                    crn = nvn
                cl.write(f"## {crn}\n")
            for ctn, ct in cr:
                if any(ln.strip() for ln in ct):
                    if ctn:
                        cl.write(f"### {ctn}\n")
                    lines = [ln.strip() for ln in ct if "code freeze date: " not in ln.lower()]
                    if not ctn and crn.lower() == nvn:
                        print("setting date")
                        for i, line in enumerate(lines):
                            if "release date: " in line.lower():
                                today = time.strftime("%Y-%m-%d")
                                lines[i] = f"Release date: {today}"
                    cl.write("\n".join(lines).replace("\n\n", "\n"))
                    cl.write("\n")
    return GitFile('CHANGELOG.md')


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
    return GitFile(file_with_version)


class GitFile():
    """Helper class for `git add`."""
    def __init__(self, path):
        self.path = path
    
    def gitadd(self):
        """run `git add`"""
        answer = subprocess.run(
            ["git", "add", self.path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf8")
        #print(answer)


class Git():
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
            if ("nothing to commit" in message):
                print("repo already up to date with new version number")
            else:
                raise Exception(f"failed to run: {message}") from err


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
    update_readme(new_version_number).gitadd()

    Git().commit(new_version_number)


if __name__ == "__main__":
    from sys import argv
    try:
        level = argv[1]
    except IndexError:
        level = "patch"
    prepare_new_release(level)

