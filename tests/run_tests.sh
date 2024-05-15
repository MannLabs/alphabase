INCLUDED_NBS=$(find ./nbdev_nbs -name "*.ipynb")
python -m pytest --nbmake $(echo $INCLUDED_NBS)


INCLUDED_NBS=$(find ./docs/nbs -name "*.ipynb" | grep -v tutorial_dev_spectral_libraries.ipynb)
python -m pytest --nbmake $(echo $INCLUDED_NBS)


INCLUDED_NBS=$(find ./nbs_tests -name "*.ipynb" | grep -v test_isotope_mp.ipynb)
python -m pytest --nbmake $(echo $INCLUDED_NBS)
