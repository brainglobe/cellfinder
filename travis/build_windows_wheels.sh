

conda create -n test-environment python=$TRAVIS_PYTHON_VERSION
source activate test-environment

python setup.py bdist_wheel
