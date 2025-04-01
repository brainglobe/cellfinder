import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import keras
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.core.classify.tools import get_model
from cellfinder.core.classify.resnet import layer_type

# Define models mapping directly here to avoid circular imports
models = {
    "18": "18-layer",
    "34": "34-layer", 
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}

class ClassificationParameters:
    """Configuration parameters for classification."""
    def __init__(
        self,
        batch_size: int = 64,
        cube_height: int = 50,
        cube_width: int = 50,
        cube_depth: int = 20,
        network_depth: layer_type = "50-layer",
        max_workers: int = 3,
    ):
        """
        Parameters
        ----------
        batch_size : int
            Size of each batch for classification
        cube_height : int
            Height of the cube in pixels
        cube_width : int
            Width of the cube in pixels
        cube_depth : int
            Depth of the cube in pixels
        network_depth : layer_type
            Depth of the neural network (e.g. "50-layer")
        max_workers : int
            Maximum number of worker processes
        """
        self.batch_size = batch_size
        self.cube_height = cube_height
        self.cube_width = cube_width
        self.cube_depth = cube_depth
        self.network_depth = network_depth
        self.max_workers = max_workers

class DataParameters:
    """Configuration parameters for input data."""
    def __init__(
        self,
        voxel_sizes: Tuple[int, int, int],
        network_voxel_sizes: Tuple[int, int, int],
        n_free_cpus: int = 2,
    ):
        """
        Parameters
        ----------
        voxel_sizes : Tuple[int, int, int]
            Voxel sizes of the input data
        network_voxel_sizes : Tuple[int, int, int]
            Voxel sizes expected by the network
        n_free_cpus : int
            Number of CPU cores to leave free
        """
        self.voxel_sizes = voxel_sizes
        self.network_voxel_sizes = network_voxel_sizes
        self.n_free_cpus = n_free_cpus

def classify_with_params(
    points: List[Cell],
    signal_array: types.array,
    background_array: types.array,
    data_parameters: DataParameters,
    classification_parameters: ClassificationParameters,
    trained_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """Classify cell candidates into cells and non-cells.

    Parameters
    ----------
    points : List[Cell]
        List of detected cell candidates to be classified
    signal_array : types.array
        The signal (e.g. channel 0) data
    background_array : types.array
        The background (e.g. channel 1) data
    data_parameters : DataParameters
        Configuration for the input data
    classification_parameters : ClassificationParameters
        Configuration for the classification process
    trained_model : Optional[os.PathLike]
        Path to a trained model
    model_weights : Optional[os.PathLike]
        Path to model weights
    callback : Callable[int], optional
        A callback function that is called during classification. Called with
        the batch number once that batch has been classified.

    Returns
    -------
    List[Cell]
        Classified cell candidates
    """
    if signal_array.ndim != 3:
        raise IOError("Signal data must be 3D")
    if background_array.ndim != 3:
        raise IOError("Background data must be 3D")

    if callback is not None:
        callbacks = [BatchEndCallback(callback)]
    else:
        callbacks = None

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(min_free_cpu_cores=data_parameters.n_free_cpus)

    start_time = datetime.now()
    logger.debug("Initialising cube generator")
    
    # Create the cube generator using the configuration parameters
    inference_generator = CubeGeneratorFromFile(
        points,
        signal_array,
        background_array,
        data_parameters.voxel_sizes,
        data_parameters.network_voxel_sizes,
        batch_size=classification_parameters.batch_size,
        cube_width=classification_parameters.cube_width,
        cube_height=classification_parameters.cube_height,
        cube_depth=classification_parameters.cube_depth,
        use_multiprocessing=False,
        workers=workers,
    )

    if trained_model and Path(trained_model).suffix == ".h5":
        print(
            "Weights provided in place of the model, "
            "loading weights into default model."
        )
        model_weights = trained_model
        trained_model = None

    model = get_model(
        existing_model=trained_model,
        model_weights=model_weights,
        network_depth=models.get(classification_parameters.network_depth.split("-")[0], "50-layer"),
        inference=True,
    )

    logger.info("Running inference")
    # in Keras 3.0 multiprocessing params are specified in the generator
    predictions = model.predict(
        inference_generator,
        verbose=True,
        callbacks=callbacks,
    )

    predictions = predictions.round()
    predictions = predictions.astype("uint16")
    predictions = np.argmax(predictions, axis=1)

    points_list = []
    # only go through the "extractable" points
    for idx, cell in enumerate(inference_generator.ordered_points):
        cell.type = predictions[idx] + 1
        points_list.append(cell)

    time_elapsed = datetime.now() - start_time
    print(
        "Classfication complete - all points done in : {}".format(time_elapsed)
    )

    return points_list

# Original function signature for backward compatibility
def main(
    points: List[Cell],
    signal_array: types.array,
    background_array: types.array,
    n_free_cpus: int,
    voxel_sizes: Tuple[int, int, int],
    network_voxel_sizes: Tuple[int, int, int],
    batch_size: int,
    cube_height: int,
    cube_width: int,
    cube_depth: int,
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    network_depth: str,
    max_workers: int = 3,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Parameters
    ----------
    callback : Callable[int], optional
        A callback function that is called during classification. Called with
        the batch number once that batch has been classified.
    """
    # Create configuration objects from individual parameters
    data_params = DataParameters(
        voxel_sizes=voxel_sizes,
        network_voxel_sizes=network_voxel_sizes,
        n_free_cpus=n_free_cpus
    )
    
    # Convert the string network_depth to the layer_type format
    layer_network_depth = models.get(network_depth, "50-layer")
    
    classification_params = ClassificationParameters(
        batch_size=batch_size,
        cube_height=cube_height, 
        cube_width=cube_width,
        cube_depth=cube_depth,
        network_depth=layer_network_depth,
        max_workers=max_workers
    )
    
    # Call the new implementation with the configuration objects
    return classify_with_params(
        points,
        signal_array, 
        background_array,
        data_params,
        classification_params,
        trained_model,
        model_weights,
        callback=callback
    )

class BatchEndCallback(keras.callbacks.Callback):
    def __init__(self, callback: Callable[[int], None]):
        self._callback = callback

    def on_predict_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self._callback(batch)