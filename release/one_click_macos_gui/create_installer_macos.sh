#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=AlphaBase.pkg
if test -f "$FILE"; then
  rm AlphaBase.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphabaseinstaller python=3.9 -y
conda activate alphabaseinstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/alphabase-1.2.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller ../pyinstaller/alphabase.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alphabase/data/*.fasta dist/alphabase/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/alphabase/Contents/Resources
cp ../logos/alpha_logo.icns dist/alphabase/Contents/Resources
mv dist/alphabase_gui dist/alphabase/Contents/MacOS
cp Info.plist dist/alphabase/Contents
cp alphabase_terminal dist/alphabase/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/alpha_logo.png Resources/alpha_logo.png
chmod 777 scripts/*

pkgbuild --root dist/alphabase --identifier de.mpg.biochem.alphabase.app --version 1.2.2 --install-location /Applications/AlphaBase.app --scripts scripts AlphaBase.pkg
productbuild --distribution distribution.xml --resources Resources --package-path AlphaBase.pkg dist/alphabase_gui_installer_macos.pkg
