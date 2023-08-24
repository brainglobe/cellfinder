from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellfinder")
except PackageNotFoundError as e:
    raise PackageNotFoundError("cellfinder package not installed") from e

# If tensorflow is not present, tools cannot be used.
# Throw an error in this case to prevent invocation of functions.
try:
    TF_VERSION = version("tensorflow")
except PackageNotFoundError as e:
    try:
        TF_VERSION = version("tensorflow-macos")
    except PackageNotFoundError as e:
        raise PackageNotFoundError(
            f"cellfinder tools cannot be invoked without tensorflow. "
            f"Please install tensorflow into your environment to use cellfinder tools. "
            f"For more information, please see "
            f"https://github.com/brainglobe/brainglobe-meta#readme."
        ) from e

__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"
