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


# If no backend is configured and installed for Keras, tools cannot be used
# Check if backend is configured. If not, set to "torch"
if not os.getenv("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "torch"

# Check backend is installed
backend = os.getenv("KERAS_BACKEND")
if backend in ["tensorflow", "jax", "torch"]:
    try:
        BACKEND_VERSION = version(backend)
    except PackageNotFoundError as e:
        raise PackageNotFoundError(
            f"{backend} set as Keras backend " f"but not installed"
        ) from e
else:
    raise PackageNotFoundError(
        "Keras backend must be one of 'torch', 'tensorflow', or 'jax'"
    )


__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"

DEFAULT_CELLFINDER_DIRECTORY = Path.home() / ".brainglobe" / "cellfinder"
