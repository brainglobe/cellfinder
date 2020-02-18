### Contributing
#### Setup
To set up a development install, please:
* Fork this repository
* Clone your fork

    `git clone https://github.com/YOUR_USERNAME/cellfinder`
    
    `cd neuro`
* Add this repository as the upstream

    `git remote add upstream https://github.com/SainsburyWellcomeCentre/cellfinder`
    
* Install an editable, development version of `cellfinder` 

    `pip install -e .[dev]`

* To keep your fork up to date:

    `git fetch upstream`
    
    `git merge upstream/master`
    
#### Pull requests
In all cases, please submit code to the main repository via a pull request. 
Upon approval, please merge via "Squash and Merge" on Github to maintain a 
clean commit history.

#### Formatting
`neuro` uses [Black](https://github.com/python/black) o ensure a consistent 
code style. Please run `black ./ -l 79 --target-version py37` before making 
any commits. To prevent any errors, it is easier to add a formatting check 
as a [pre-commit hook](https://www.atlassian.com/git/tutorials/git-hooks). 
E.g. on linux by adding this to your `.git/hooks/pre-commit`:

    black ./ -l 79 --target-version py37 --check || exit 1
    
#### Testing
`cellfinder` uses [pytest](https://docs.pytest.org/en/latest/) for testing. Please 
try to ensure that all functions are tested in `tests/tests/test_unit` and 
all workflows/command-line tools are tested in `tests/tests/test_integration`.

Some tests may 
take a long time (e.g. those requiring tensorflow if you don't have a GPU). 
These tests should be marked with `@pytest.mark.slow`, e.g.:

```python
import pytest
@pytest.mark.slow
def test_something_slow():
    slow_result = run_slow_processes()
    assert slow_result == expected_slow_thing
```

During development, these "slow" tests can be skipped by running `
pytest -m "not slow"`.

#### Travis
All commits & pull requests will be build by [Travis](https://travis-ci.com). 
To ensure there are no issues, please check that it builds: `pip install .` 
and run all of the tests: `pytest` before commiting changes. 


#### Releases
Travis will automatically release any tagged commit on the master branch. 
Hence to release a new version of cellfinder, use either GitHub, or the git 
CLI to tag the relevant commit and push to master.


#### Documentation
Documentation is built using Sphinx with Markdown (rather than 
reStructuredText). Please edit (or create new) `.md` files in the appropriate 
directory in `cellfinder/doc_build`. The organisation of these files is then 
defined in `index.rst`.

Please ensure that:
* Any new changes are added to the release notes
(`doc_build/main/about/release_notes.md`)
* All command-line functions are fully documented, including an explanation 
of all arguments, and example syntax.

To build the documentation (assuming you installed cellfinder with 
`pip install -e .[dev]` to install the dependencies):

```bash
cd doc_build
make html
```

Prior to commmiting to master, ensure that contents of 
`doc_build/_build/html/` is copied to `cellfinder/docs` for 
hosting with [github pages](https://adamltyson.github.io/cellfinder/index.html).

This can be done automatically with a 
[pre-commit hook](https://www.atlassian.com/git/tutorials/git-hooks). An 
example is 
[here](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/pre-commit). 


##### Dependencies
The code in the cellfinder repo should be primarily for cell detection. Any 
changes to the registration should be added to 
[amap](https://github.com/sainsburywellcomecentre/amap-python), any 
visualisation should be added to 
[neuro](https://github.com/sainsburywellcomecentre/neuro) and any general 
tools to [imlib](https://github.com/adamltyson/imlib).


##### File paths
All file paths should be defined in `cellfinder.tools.prep.Paths`. Any 
intermediate file paths, (i.e. those which are not of interest to the typical 
end user) should be prefixed with `tmp__`. These should then be cleaned up as 
soon as possible after generation. 

##### Conventions
Cellfinder has recently (2019-08-02) dropped support for Python 3.5. Following 
this, a number of new python features will be adopted throughout.

* [pathlib](https://realpython.com/python-pathlib/) conventions 
(rather then `os.path`).
* [f-strings](https://realpython.com/python-f-strings/) 
(rather than `.format()` or using the old `%` operator). 

In all cases, please aim to replace old calls to `os.path` or `.format()` 
with pathlib object methods and f-strings.
