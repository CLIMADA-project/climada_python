#!/bin/bash -e

source activate climada_env
conda env update --file requirements/env_developer.yml

make lint
make test
conda deactivate
