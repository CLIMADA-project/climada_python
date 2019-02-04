#!/bin/bash -e

source activate climada_env
conda env update --file requirements/env_developer.yml
make lint
make integ_test
source deactivate
