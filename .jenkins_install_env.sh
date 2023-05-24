#!/bin/bash -e

mamba remove --name climada_env --all
mamba env create -f requirements/env_climada.yml --name climada_env

source activate climada_env
python -m pip install -e "./[test]"
make install_test
conda deactivate
