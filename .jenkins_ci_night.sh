#!/bin/bash -e

conda activate climada_env
conda env update --file requirements/env_developer.yml
make lint
make test
conda deactivate
