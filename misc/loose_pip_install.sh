conda create -n alphabase python=3.8 -y
conda activate alphabase
pip install -e '../.[development]' -U
alphabase
conda deactivate
