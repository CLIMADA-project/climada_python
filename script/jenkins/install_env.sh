#!/bin/bash -e

mamba env remove -n climada_env -y
mamba create -n climada_env python=3.11 -y
mamba env update -n climada_env -f requirements/env_climada.yml

source activate climada_env
python -m pip install -e "./[dev]"

make install_test

conda deactivate
