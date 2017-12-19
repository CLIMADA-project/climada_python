#! /usr/bin/env bash 
set -e

# activate climada environment
source /Users/aznarsig/anaconda3/envs/climada/bin/activate climada

# run all climada tests
#PYTHONPATH=. python -m coverage run --include climada/entity/assets.py test_xmlrunner.py
PYTHONPATH=. python -m coverage run test_xmlrunner.py
#coverage run test_xmlrunner.py

# analize coverage
python -m coverage xml -o coverage.xml
python -m coverage html -d coverage

# run static code analysis
pylint -ry --load-plugins=pylint.extensions.mccabe --extension-pkg-whitelist=numpy --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" climada > pylint.log || true

