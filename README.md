# AlphaBase

![Pip installation](https://github.com/MannLabs/alphabase/workflows/Default%20installation%20and%20tests/badge.svg)
![PyPi releases](https://github.com/MannLabs/alphabase/workflows/Publish%20on%20PyPi%20and%20release%20on%20GitHub/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/alphabase/badge/?version=latest)](https://alphabase.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/alphabase)](https://pypi.org/project/alphabase)
[![pip downloads](https://img.shields.io/pypi/dm/alphabase?color=blue&label=pip%20downloads)](https://pypi.org/project/alphabase)
![Python](https://img.shields.io/pypi/pyversions/alphabase)

AlphaBase provides all basic python functionalities for AlphaPept
ecosystem from the [Mann Labs at the Max Planck Institute of
Biochemistry](https://www.biochem.mpg.de/mann) and the [University of
Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/). To enable
all hyperlinks in this document, please view it at
[GitHub](https://github.com/MannLabs/alphabase). For documentation,
please see [readthedocs](https://alphabase.readthedocs.io/en/latest).

- [**About**](#about)
- [**License**](#license)
- [**Installation**](#installation)
  - [**Pip installer**](#pip)
  - [**Developer installer**](#developer)
- [**Usage**](#usage)
- [**Troubleshooting**](#troubleshooting)
- [**Citations**](#citations)
- [**How to contribute**](#how-to-contribute)
- [**Changelog**](#changelog)

------------------------------------------------------------------------

## About

The infrastructure package of AlphaX ecosystem for MS proteomics. It was first published with AlphaPeptDeep, see [Citations](#citations).

### Packages built upon AlphaBase

- [AlphaPeptDeep](https://github.com/MannLabs/alphapeptdeep): deep learning framework for proteomics.
- [AlphaRaw](https://github.com/MannLabs/alpharaw): raw data reader for different vendors.
- [AlphaDIA](https://github.com/MannLabs/alphadia): DIA search engine.
- [PeptDeep-HLA](https://github.com/MannLabs/peptdeep-hla): personalized HLA-binding peptide prediction.
- [AlphaViz](https://github.com/MannLabs/alphaviz): visualization for MS-based proteomics.
- [AlphaQuant](https://github.com/MannLabs/alphaquant): quantification for MS-based proteomics.

------------------------------------------------------------------------

## Citations

Wen-Feng Zeng, Xie-Xuan Zhou, Sander Willems, Constantin Ammar, Maria Wahle, Isabell Bludau, Eugenia Voytik, Maximillian T. Strauss & Matthias Mann. AlphaPeptDeep: a modular deep learning framework to predict peptide properties for proteomics. Nat Commun 13, 7238 (2022). https://doi.org/10.1038/s41467-022-34904-3

------------------------------------------------------------------------

## License

AlphaBase was developed by the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann) and the [University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/) and is
freely available with an [Apache License](LICENSE.txt). External Python
packages (available in the [requirements](requirements) folder) have
their own licenses, which can be consulted on their respective websites.

------------------------------------------------------------------------

## Installation

AlphaBase can be installed and used on all major operating systems
(Windows, macOS and Linux). There are two different types of
installation possible:

- [**Pip installer:**](#pip) Choose this installation if you want to use
  AlphaBase as a Python package in an existing Python 3.8 environment
  (e.g. a Jupyter notebook).
- [**Developer installer:**](#developer) Choose this installation if you
  are familiar with [conda](https://docs.conda.io/en/latest/) and
  Python. This installation allows access to all available features of
  AlphaBase and even allows to modify its source code directly.
  Generally, the developer version of AlphaBase outperforms the
  precompiled versions which makes this the installation of choice for
  high-throughput experiments.

### Pip

AlphaBase can be installed in an existing Python 3.8 environment with a
single `bash` command. *This `bash` command can also be run directly
from within a Jupyter notebook by prepending it with a `!`*:

``` bash
pip install alphabase
```

Installing AlphaBase like this avoids conflicts when integrating it in
other tools, as this does not enforce strict versioning of dependencies.
However, if new versions of dependencies are released, they are not
guaranteed to be fully compatible with AlphaBase. While this should only
occur in rare cases where dependencies are not backwards compatible, you
can always force AlphaBase to use dependency versions which are known to
be compatible with:

``` bash
pip install "alphabase[stable]"
```

NOTE: You might need to run `pip install -U pip` before installing
AlphaBase like this. Also note the double quotes `"`.
If you are using the `quant_reader` module, it is advisable to add the
`dask-stable` or `dask` extras to speed up processing large files.
You need to install the `hdf` extra option of the package to be able to read alphapept protein group matrices in hdf format.

For those who are really adventurous, it is also possible to directly
install any branch (e.g. `@main`) with any extras
(e.g. `#egg=alphabase[stable,development]`) from GitHub with e.g.

``` bash
pip install "git+https://github.com/MannLabs/alphabase.git@main#egg=alphabase[stable,development]"
```

### Developer

AlphaBase can also be installed in editable (i.e. developer) mode with a
few `bash` commands. This allows to fully customize the software and
even modify the source code to your specific needs. When an editable
Python package is installed, its source code is stored in a transparent
location of your choice. While optional, it is advised to first (create
and) navigate to e.g. a general software folder:

``` bash
mkdir ~/folder/where/to/install/software
cd ~/folder/where/to/install/software
```

***The following commands assume you do not perform any additional `cd`
commands anymore***.

Next, download the AlphaBase repository from GitHub either directly or
with a `git` command. This creates a new AlphaBase subfolder in your
current directory.

``` bash
git clone https://github.com/MannLabs/alphabase.git
```

For any Python package, it is highly recommended to use a separate
[conda virtual environment](https://docs.conda.io/en/latest/), as
otherwise *dependency conflicts can occur with already existing
packages*.

``` bash
conda create --name alphabase python=3.9 -y
conda activate alphabase
```

Finally, AlphaBase and all its [dependencies](requirements) need to be
installed. To take advantage of all features and allow development (with
the `-e` flag), this is best done by also installing the [development
dependencies](requirements/requirements_development.txt) instead of only
the [core dependencies](requirements/requirements.txt):

``` bash
pip install -e "./alphabase[development]"
```

By default this installs loose dependencies (no explicit versioning),
although it is also possible to use stable dependencies
(e.g. `pip install -e "./alphabase[stable,development]"`).

***By using the editable flag `-e`, all modifications to the [AlphaBase
source code folder](alphabase) are directly reflected when running
AlphaBase. Note that the AlphaBase folder cannot be moved and/or renamed
if an editable version is installed. In case of confusion, you can
always retrieve the location of any Python module with e.g. the command
`import module` followed by `module.__file__`.***

------------------------------------------------------------------------

## Usage

TODO

------------------------------------------------------------------------

## Troubleshooting

In case of issues, check out the following:

- [Issues](https://github.com/MannLabs/alphabase/issues): Try a few
  different search terms to find out if a similar problem has been
  encountered before
- [Discussions](https://github.com/MannLabs/alphabase/discussions):
  Check if your problem or feature requests has been discussed before.

------------------------------------------------------------------------

## How to contribute

If you like this software, you can give us a
[star](https://github.com/MannLabs/alphabase/stargazers) to boost our
visibility! All direct contributions are also welcome. Feel free to post
a new [issue](https://github.com/MannLabs/alphabase/issues) or clone the
repository and create a [pull
request](https://github.com/MannLabs/alphabase/pulls) with a new branch.
For an even more interactive participation, check out the
[discussions](https://github.com/MannLabs/alphabase/discussions) and the
[the Contributors License Agreement](misc/CLA.md).

### Notes for developers

#### 1. Code Structure
While AlphaBase offers an object-oriented interface, algorithms for manipulating data should be implemented in a functional way and called from class methods. This allows the functions to be reused without instatiating a class.

#### 2. DataFrame Handling
- Return DataFrames in the same order as they were passed
- Minimize in-place modifications of DataFrames. Mention them explicitly in the docstring
- Implement low-level functions that operate on numpy arrays and return arrays. Use higher-level functions to assign array results to DataFrames

#### 3. Data Assumptions
Avoid making assumptions about:
- Precursor ordering by `nAA`
- Fragment indices ordering (e.g., `frag_start_idx`)
- Continuity of `frag_start_idx` where `frag_start_idx[i+1] == frag_stop_idx[i]`
- All fragments being assigned to a precursor

Assumptions are only permitted for low-level or optimized functions and should be documented in the docstring.

#### 3. Optimization Strategy
When performance optimization is needed:
1. Implement the general solution first
2. Add optimized versions for special cases for refined precursor df or order `nAA`
3. Check conditions at runtime to use optimized versions when applicable

#### 4. Code Quality
- Include python type hints
- Include docstrings in numpy style (see [numpy docstring example](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy))


#### 5. pre-commit hooks
It is highly recommended to use the provided pre-commit hooks, as the CI pipeline enforces all checks therein to
pass in order to merge a branch.

The hooks need to be installed once by
```bash
pre-commit install
```
You can run the checks yourself using:
```bash
pre-commit run --all-files
```

#### 6. Tagging of Pull Requests
In order to have release notes automatically generated, pull requests need to be tagged with labels.
The following labels are used (should be safe-explanatory):
`breaking-change`, `bug`, `enhancement`.

#### 7. Release a new version
This package uses a shared release process defined in the
[alphashared](https://github.com/MannLabs/alphashared) repository. Please see the instructions
[there](https://github.com/MannLabs/alphashared/blob/reusable-release-workflow/.github/workflows/README.md#release-a-new-version).


------------------------------------------------------------------------

## Changelog

For a full overview of the changes made in each version see [CHANGELOG.md](CHANGELOG.md) (until version 1.1.0) and the github release notes (from >1.1.0).
