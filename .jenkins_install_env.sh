#!/bin/sh -xe

set +e

conda remove --name climada_env --all
conda env create -f requirements/env_climada.yml --name climada_env

. activate climada_env
python3 tests_runner.py
. deactivate
