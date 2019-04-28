#!/bin/bash -e

source activate climada_env
make data_test
conda deactivate
