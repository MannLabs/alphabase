conda create -n alphabase_pip_test python=3.9 -y
conda activate alphabase_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "alphabase[stable]"
alphabase
conda deactivate
