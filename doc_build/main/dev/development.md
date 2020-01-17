### Developer notes
#### Code organisation

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
 

#### Travis
All commits & pull requests will be build by [Travis](https://travis-ci.com). 
To ensure there are no issues, please check that it builds: `pip install .` 
and run all of the tests: `pytest` before commiting changes. Some tests may 
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

#### Formatting
To ensure a consistent code style, please use
 [Black](https://github.com/python/black) before commiting changes. 
 Please use the syntax: `black ./ -l 79 --target-version py36`
 

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
make latexpdf
```

Prior to commmiting to master, ensure that 
`doc_build/_build/latex/cellfinder.pdf` is copied to `cellfinder/` and that 
the contents of `doc_build/_build/html/` is copied to `cellfinder/docs` for 
hosting with [github pages](https://adamltyson.github.io/cellfinder/index.html).

This can be done automatically with a 
[pre-commit hook](https://www.atlassian.com/git/tutorials/git-hooks). An 
example is [here](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/pre-commit). 


#### Dependencies
Any dependencies in the `dev` branch will be checked by 
[dependabot](https://dependabot.com/), and a pull request will be generated to 
update which are outdated. Please only merge these if all tests are passing, 
and you are confident that there will be no issues.

#### Misc
**Configuration files**

Any configuration files that cellfinder edits with local information (e.g. 
atlas installation location) should be in the `.gitignore` file. Please ensure 
that any custom files are not commited to the repository.
