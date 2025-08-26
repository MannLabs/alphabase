#!/bin/bash
### Install the package with a given type in a defined conda environment with a define python version,
### and call it to check if it works
### example usage:
### ./pip_install.sh stable my_env 3.9
set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alphabase}
PYTHON_VERSION=${3:-3.9}
OS=${4:-nan}

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[${INSTALL_TYPE}]"
fi

# pytables has known issues on MacOS for pg-readers - install from conda
# https://github.com/PyTables/PyTables/issues/219#issuecomment-24117053
if [[ "$OS" = "macOS-latest" || "$OS" = "macos-latest-xlarge" ]]; then
  conda install -n $ENV_NAME -c conda-forge pytables -y
fi

# print pip environment for reproducibility
conda run -n $ENV_NAME --no-capture-output pip freeze

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e "../.$INSTALL_STRING"
conda run -n $ENV_NAME --no-capture-output python -c "import alphabase; print('OK')" -v
