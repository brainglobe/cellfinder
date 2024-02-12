import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

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
# Check backend is configured (default: JAX)
# do not use default in getenv: any changes to environment after importing
# os module are not picked up except if directly modifying the dict
if not os.getenv("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "jax"
    warnings.warn("Keras backend not configured, automatically set to JAX")

# Check backend is installed
backend = os.getenv("KERAS_BACKEND")
if backend in ["tensorflow", "jax", "torch"]:
    try:
        backend_package = "tf-nightly" if backend == "tensorflow" else backend
        BACKEND_VERSION = version(backend_package)

        warnings.warn(f"Using Keras with {backend} backend")

    except PackageNotFoundError as e:
        raise PackageNotFoundError(
            f"{backend}, ({backend_package}) set as Keras backend "
            f"but not installed"
        ) from e
else:
    raise PackageNotFoundError(
        f"Keras backend must be one of 'tensorflow', "
        f"'jax', or 'torch' (not {backend})."
    )

# If TF is installed but backend not set to TF, raise an error
tf_spec = find_spec("tensorflow")
if tf_spec:
    if tf_spec.name != backend:
        raise ImportError(
            f"Tensorflow package installed"
            f"but not set as Keras backend ({backend})"
        )  # replace by ImportWarning?


__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"
