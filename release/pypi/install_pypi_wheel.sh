conda create -n alphabase_pip_test python=3.9 -y
conda activate alphabase_pip_test
pip install "alphabase[stable]"
alphabase
conda deactivate
