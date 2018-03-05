#! /usr/bin/env bash

# Delete previous env folder containing the environment
conda remove --name climada_env --all

# Create new environment
conda create --name climada_env python=3.6
# Install packages
source activate climada_env
pip install -r requirements.txt

# for mac os x: use TkAgg backend
sed -i '' '/macosx/d' ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
echo "backend : TkAgg" >> ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc

# Execute tests to verify installation
PYTHONPATH=. python3 unit_tests.py

source deactivate
