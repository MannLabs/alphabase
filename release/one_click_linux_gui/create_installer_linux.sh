#!bash

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphabase_installer python=3.8 -y
conda activate alphabase_installer

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_linux_gui
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/alphabase-0.3.1-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller ../pyinstaller/alphabase.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alphabase/data/*.fasta dist/alphabase/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/AlphaBase_gui_installer_linux/usr/local/bin
mv dist/AlphaBase dist/AlphaBase_gui_installer_linux/usr/local/bin/AlphaBase
mkdir dist/AlphaBase_gui_installer_linux/DEBIAN
cp control dist/AlphaBase_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/AlphaBase_gui_installer_linux/
