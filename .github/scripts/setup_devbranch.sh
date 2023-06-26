#!/usr/bin/bash

set -e

git config --global user.name 'climada'
git config --global user.email 'test.climada@gmail.com'

git fetch --unshallow || echo cannot \"git fetch --unshallow \"
git checkout develop
git pull

git checkout origin/main \
    setup.py \
    doc/misc/README.md \
    CHANGELOG.md \
    */_version.py

release=`python .github/scripts/setup_devbranch.py`

git add \
    setup.py \
    doc/misc/README.md \
    CHANGELOG.md \
    */_version.py

git commit -m "setup develop branch for $release"

git push || echo cannot \"git push\"
git checkout main
