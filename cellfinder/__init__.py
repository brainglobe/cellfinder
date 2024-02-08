from importlib.metadata import PackageNotFoundError, version

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
        f"Please install Keras into your environment to use cellfinder tools. "
        f"For more information, please see "
        f"https://github.com/brainglobe/brainglobe-meta#readme."
    ) from e

# Configure Keras backend:
# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
try:
    import os

    # if environment variable does not exist, assign TF
    # options: "torch" "jax", "tensorflow"
    if not os.getenv("KERAS_BACKEND"):
        os.environ["KERAS_BACKEND"] = "tensorflow"


except PackageNotFoundError as e:
    raise PackageNotFoundError("error setting up Keras backend") from e


__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"
