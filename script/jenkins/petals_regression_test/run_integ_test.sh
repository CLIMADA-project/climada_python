#!/bin/bash
export PATH=$PATH:$CONDAPATH
mamba env update -n climada_env -f ~/jobs/petals_install_env/workspace/requirements/env_climada.yml

source activate climada_env

REGTESTENV=~/jobs/petals_compatibility/petals_env
BRANCH=$1
echo ::: $REGTESTENV/$BRANCH
PETALS_DIR=`test -e $REGTESTENV/$BRANCH && cat $REGTESTENV/$BRANCH || echo ~/jobs/petals_branches/branches/develop/workspace`

python -m venv --system-site-packages tvenv
source tvenv/bin/activate

pip install -e $PETALS_DIR

cp $PETALS_DIR/climada.conf climada.conf
python script/jenkins/set_config.py test_directory $PETALS_DIR/climada_petals

PYTHONPATH=.:$PYTHONPATH pytest --junitxml=tests_xml/tests.xml $PETALS_DIR/climada_petals

git checkout climada.conf

deactivate
#rm -r tvenv
