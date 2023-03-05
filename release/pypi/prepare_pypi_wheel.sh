cd ../..
conda create -n alphabase_pypi_wheel python=3.9
conda activate alphabase_pypi_wheel
pip install twine
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
twine check dist/*
conda deactivate
