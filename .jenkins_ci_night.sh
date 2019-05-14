#!/bin/bash -e

source activate climada_env

pip list
conda list

conda env update --file requirements/env_developer.yml

pip list
conda list

make lint
make test
conda deactivate
