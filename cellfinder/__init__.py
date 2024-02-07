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
# https://keras.io/getting_started/intro_to_keras_for_engineers/
# https://github.com/keras-team/keras/blob/5bc8488c0ea3f43c70c70ebca919093cd56066eb/keras/backend/config.py#L263
try:
    import os

    # check if environment variable exists?
    os.environ["KERAS_BACKEND"] = "jax"  # "torch" "jax", "tensorflow"

except PackageNotFoundError as e:
    raise PackageNotFoundError("error setting up Keras backend") from e


__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"
