#pip install --upgrade pip
pip install setuptools --upgrade
pip install -e .[dev]
black .\ -l 79 --target-version py37 --check
cellfinder -h
cellfinder_download --atlas allen_2017_100um
pytest --cov=cellfinder