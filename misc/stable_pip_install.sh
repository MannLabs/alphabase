conda create -n alphabase python=3.9 -y
conda activate alphabase
pip install -e '../.[stable,development]' -U
alphabase
conda deactivate
