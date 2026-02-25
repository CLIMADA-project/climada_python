#!/bin/bash -e

mamba env remove -n climada_env -y
mamba create -n climada_env python=3.11 -y
mamba install -n climada_env -f requirements/env_climada.yml python=3.11 -y

source activate climada_env
python -m pip install -e "./[dev]"

make install_test

conda deactivate
