#!/bin/bash -e

mamba remove --name climada_env --all
mamba create -n climada_env python=3.11
mamba env update -n climada_env -f requirements/env_climada.yml

source activate climada_env
python -m pip install -e "./[test]"

make install_test

conda deactivate
