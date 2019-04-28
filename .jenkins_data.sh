#!/bin/bash -e

conda activate climada_env
make data_test
conda deactivate
