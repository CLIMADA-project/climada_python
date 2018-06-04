#!/bin/sh
. activate climada_env
conda env update --file requirements/env_developer.yml
make lint
make unit_test
. deactivate
