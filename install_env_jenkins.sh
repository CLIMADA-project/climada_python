#! /usr/bin/env bash

# Virtual environment using conda
# Delete previous env folder containing the environment
conda remove --name climada_env --all

# Create new environment
conda create --name climada_env python=3.6

# Install packages in new environment
source activate climada_env
python setup.py develop

# for mac os x: install python.app: Proxy on os x letting python libraries hook into the gui event loop
conda install python.app

echo "Installation completed in environment climada_env. Execute tests to check installation."
PYTHONPATH=. python3 unit_tests.py
PYTHONPATH=. python3 integ_tests.py

source deactivate
