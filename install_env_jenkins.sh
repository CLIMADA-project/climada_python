#! /usr/bin/env bash
set -e

# Install the environment
PYENV_HOME=$WORKSPACE/.pyenv/
# Delete previously built virtualenv
if [ -d $PYENV_HOME ]; then
    rm -rf $PYENV_HOME
fi
# Create virtualenv and install necessary packages
conda create -n climada_jenkins --file requirements.txt
