conda create -n alphabase_pip_test python=3.8 -y
conda activate alphabase_pip_test
pip install "alphabase[stable]"
alphabase
conda deactivate
