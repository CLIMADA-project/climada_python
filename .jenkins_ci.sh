#!/bin/bash -e

conda activate climada_env
make lint
make unit_test
conda deactivate
