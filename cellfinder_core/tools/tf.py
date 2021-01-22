import logging
import tensorflow as tf


def allow_gpu_memory_growth():
    """
    If a gpu is present, prevent tensorflow from using all the memory straight
    away. Allows multiple processes to use the GPU (and avoid occasional
    errors on some systems) at the cost of a slight performance penalty.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        logging.debug("Allowing GPU memory growth")
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logging.debug(
                f"{len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs"
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        logging.debug("No GPUs found, using CPU.")


def set_tf_threads(max_threads):
    """
    Limit the number of threads that tensorflow uses
    :param max_threads: Maximum number of threads to use
    :return:
    """
    logging.debug(
        f"Setting maximum number of threads for tensorflow "
        f"to: {max_threads}"
    )

    # If statements are for testing. If tf is initialised, then setting these
    # parameters throws an error
    if tf.config.threading.get_inter_op_parallelism_threads() != 0:
        tf.config.threading.set_inter_op_parallelism_threads(max_threads)
    if tf.config.threading.get_intra_op_parallelism_threads() != 0:
        tf.config.threading.set_intra_op_parallelism_threads(max_threads)
