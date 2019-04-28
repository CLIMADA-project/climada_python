#!/bin/bash -e

source activate climada_env
make lint
make unit_test
conda deactivate
