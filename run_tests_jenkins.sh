#! /usr/bin/env bash 
set -e

# Load previously installed climada_jenkins environment
source activate climada_jenkins

# run all climada tests
PYTHONPATH=. python -m coverage run test_xmlrunner.py

# analize coverage
python -m coverage xml -o coverage.xml
python -m coverage html -d coverage

# run static code analysis
pylint -ry --load-plugins=pylint.extensions.mccabe --extension-pkg-whitelist=numpy --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" climada > pylint.log || true

