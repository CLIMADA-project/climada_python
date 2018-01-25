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
sed -i '' '/macosx/d' env/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
echo "backend : TkAgg" >> env/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc

# Execute tests to verify installation
PYTHONPATH=. python3 unit_tests.py

source deactivate
