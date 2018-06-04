#!/bin/bash -e

source activate climada_env
make lint
make unit_test
source deactivate
