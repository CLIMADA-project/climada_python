#!/bin/bash -e

conda update -n base -c defaults conda
conda remove --name climada_env --all
conda env create -f requirements/env_climada.yml --name climada_env

source activate climada_env
make install_test
conda deactivate
