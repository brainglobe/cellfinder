#wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
#bash miniconda.sh -b -p $HOME/miniconda
#export PATH="$HOME/miniconda/bin:$PATH"
#hash -r
#conda config --set always_yes yes --set changeps1 no
#conda info -a
#conda create -n test-environment python=$TRAVIS_PYTHON_VERSION
#source activate test-environment
#pip install -e .[dev]
#conda info -a
#black ./ -l 79 --check
#cellfinder -h
#cellfinder_download
#pytest --cov=cellfinder
ls