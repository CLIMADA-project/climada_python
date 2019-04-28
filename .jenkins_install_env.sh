#!/bin/bash -e

conda remove --name climada_env --all
conda env create -f requirements/env_climada.yml --name climada_env

source activate climada_env
python3 tests_install.py report
conda deactivate
