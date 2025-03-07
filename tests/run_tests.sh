# TODO make tutorial_dev_spectral_libraries.ipynb work
DOCS_NBS=$(find ../docs/nbs -name "*.ipynb" | grep -v tutorial_dev_spectral_libraries.ipynb)

# TODO make test_isotope_mp.ipynb work
# Note: multiprocessing in ipynb sometimes suspended on some versions of Windows, ignore the
# corresponding notebook(s) if this occurs again
# INCLUDED_NBS=$(find ../nbs_tests -name "*.ipynb" | grep -v test_isotope_mp.ipynb)

# we want notebook tests running, even if the first stage of pytest fails
set +e
python -m pytest
pytest_exit_code=$?
set -e

TEST_NBS=$(find ../nbs_tests -name "*.ipynb")
TUTORIAL_NBS=$(find ../docs/tutorials -name "*.ipynb")
ALL_NBS=$(echo $DOCS_NBS$'\n'$TEST_NBS$'\n'$TUTORIAL_NBS)

python -m pytest --nbmake $(echo $ALL_NBS)

if [ $pytest_exit_code -ne 0 ]; then
    echo pytest failed
    exit $pytest_exit_code
fi
