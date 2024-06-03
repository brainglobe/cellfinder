import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Check cellfinder is installed
try:
    __version__ = version("cellfinder")
except PackageNotFoundError as e:
    raise PackageNotFoundError("cellfinder package not installed") from e

# If Keras is not present, tools cannot be used.
# Throw an error in this case to prevent invocation of functions.
try:
    KERAS_VERSION = version("keras")
except PackageNotFoundError as e:
    raise PackageNotFoundError(
        f"cellfinder tools cannot be invoked without Keras. "
        f"Please install Keras with a backend into your environment "
        f"to use cellfinder tools. "
        f"For more information on Keras backends, please see "
        f"https://keras.io/getting_started/#installing-keras-3."
        f"For more information on brainglobe, please see "
        f"https://github.com/brainglobe/brainglobe-meta#readme."
    ) from e


# Set the Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__license__ = "BSD-3-Clause"

DEFAULT_CELLFINDER_DIRECTORY = Path.home() / ".brainglobe" / "cellfinder"
