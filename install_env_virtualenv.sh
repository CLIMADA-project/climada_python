#! /usr/bin/env bash

# Install virtualenv if needed
#python3 -m pip install --user virtualenv

# Delete previous env folder containing the environment
rm -rf env/
# Create new envirnment
python3 -m virtualenv env

# Install the environment
source activate env
pip install -r requirements.txt

# for mac os x: use TkAgg backend
sed -i '' '/macosx/d' ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib-2.1.2-py3.6-macosx-10.7-x86_64.egg/matplotlib/mpl-data/matplotlibrc
echo "backend : TkAgg" >> ../../../.conda/envs/climada_env/lib/python3.6/site-packages/matplotlib-2.1.2-py3.6-macosx-10.7-x86_64.egg/matplotlib/mpl-data/matplotlibrc

# Execute tests to verify installation
PYTHONPATH=. python3 unit_tests.py

source deactivate
