#! /usr/bin/env bash

# Virtual environment using conda
# Delete previous env folder containing the environment
conda remove --name climada_env --all

# Create new environment
conda create --name climada_env python=3.6

# Install packages in new environment
source activate climada_env
python setup.py develop

# for mac os x: use TkAgg backend
sed -i '' '/macosx/d' ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib-2.2.0rc1-py3.6-macosx-10.7-x86_64.egg/matplotlib/mpl-data/matplotlibrc
echo "backend : TkAgg" >> ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib-2.2.0rc1-py3.6-macosx-10.7-x86_64.egg/matplotlib/mpl-data/matplotlibrc

echo "Installation completed in environment climada_env. Execute tests to check installation."
PYTHONPATH=. python3 unit_tests.py

source deactivate
