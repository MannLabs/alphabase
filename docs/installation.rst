Installation
============

AlphaBase can be installed and used on all major operating systems (Windows, MacOS and Linux). There are two different types of installation possible:


Users
-----

Users can install AlphaBase as a Python package in an existing Python environment.

::

    ## Optional: create a python environment
    # conda create -n alphabase python=3.9 -y && conda activate alphabase

    pip install "alphabase[mzml]"

To enforce stringent dependencies (recommended), you can install the stable version of AlphaBase

::

    pip install "alphabase[stable,mzml-stable]"

The "mzml(-stable)" extra can be omitted if alphabase does not need to handle mzml files.

Development version
-------------------
For development, clone the latest version from GitHub to an appropriate location on your personal device and install an editable version:

::

    ## Optional: Create development environment
    # conda create -n alphabase_dev python=3.9 -y && conda activate alphabase_dev

    # Clone repository
    git clone https://github.com/MannLabs/alphabase.git
    cd alphabase

    # Install editable development version
    pip install -e ".[mzml,development]"
