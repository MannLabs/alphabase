#!python


__project__ = "alphabase"
__version__ = "1.0.1"
__license__ = "Apache"
__description__ = "An infrastructure Python package of the AlphaX ecosystem"
__author__ = "Mann Labs"
__author_email__ = "jalew188@gmail.com"
__github__ = "https://github.com/MannLabs/alphabase"
__keywords__ = [
    "bioinformatics",
    "software",
    "AlphaX ecosystem",
]
__python_version__ = ">=3.8"
__classifiers__ = [
    # "Development Status :: 1 - Planning",
    # "Development Status :: 2 - Pre-Alpha",
    # "Development Status :: 3 - Alpha",
    # "Development Status :: 4 - Beta",
    "Development Status :: 5 - Production/Stable",
    # "Development Status :: 6 - Mature",
    # "Development Status :: 7 - Inactive"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "alphabase=alphabase.cli:run",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "Mann Labs at CPR": "https://www.cpr.ku.dk/research/proteomics/mann/",
    "GitHub": __github__,
    "Docs": "https://alphabase.readthedocs.io/en/latest/",
    "PyPi": "https://pypi.org/project/alphabase/",
}
__extra_requirements__ = {
    "development": "requirements_development.txt",
}
