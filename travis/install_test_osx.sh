wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda info -a
conda create -n test-environment python=$TRAVIS_PYTHON_VERSION
source activate test-environment
pip install -e .[dev]
conda info -a
black ./ -l 79 --target-version py37 --check
cellfinder -h
brainglobe install -a allen_mouse_50um
pytest --cov=cellfinder