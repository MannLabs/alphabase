current_version=$(grep "__version__" ../alphabase/__init__.py | cut -f3 -d ' ' | sed 's/"//g')
current_version_as_regex=$(echo $current_version | sed 's/\./\\./g')
conda create -n version_check python=3.9 -y
conda activate version_check
set +e
already_on_pypi=$(pip install alphabase== 2>&1 | grep -c "$current_version_as_regex")
set -e
conda deactivate
if [ $already_on_pypi -ne 0 ]; then
  echo "Version is already on PyPi"
  exit 1
fi
