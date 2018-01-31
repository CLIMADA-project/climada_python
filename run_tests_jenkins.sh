#! /usr/bin/env bash 
set -e

# Load previously installed climada_jenkins environment
source ../../../.conda/envs/climada_env/bin/activate climada_env

# run all climada tests
PYTHONPATH=. python -m coverage run -a unit_tests.py
PYTHONPATH=. python -m coverage run -a integ_tests.py

# analize coverage
python -m coverage xml -o coverage.xml
python -m coverage html -d coverage

# run static code analysis
pylint -ry --rcfile=./pylint_conf.rc climada > pylint.log || true

