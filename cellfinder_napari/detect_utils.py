from dataclasses import dataclass, asdict
from pathlib import Path

import napari
from cellfinder_core.main import main as cellfinder_run
from imlib.cells.cells import Cell
from napari.qt.threading import thread_worker

from cellfinder_napari.utils import cells_to_array

@dataclass
class InputContainer:
    """Base data class to group inputs into a container, with persistent defaults"""
    persistent_defaults = {} # some elements need to be remembered without having to create a new instance, for reset button

    @classmethod
    def numerical_widget(cls, key, custom_label=None, step=0.1, min=None, max=None) -> dict:
        """Provide info to populate a numerical widget with the class's persistent defaults."""
        # TODO: maybe some assertions like either both or neither min/max==None and cls.defaults[key] is numerical.
        if custom_label is None:
            label = key.replace('_', ' ').capitalize()
        else:
            label = custom_label
        if min is None and max is None:
            return dict(value=cls.persistent_defaults[key], label=label, step=step)
        else:
            return dict(value=cls.persistent_defaults[key], label=label, step=step, min=min, max=max)

    def as_core_arguments(self) -> dict:
        """Allow customisation of how inputs are passed to cellfinder-core"""
        return asdict(self)


@dataclass
class DataInputs(InputContainer):
    """Container for image-related ("Data") inputs."""
    signal_array : napari.layers.Image
    background_array: napari.layers.Image
    voxel_size_z : float
    voxel_size_y : float
    voxel_size_x : float

    persistent_defaults = dict(
        voxel_size_z=5,
        voxel_size_y=2,
        voxel_size_x=2
    )

    def as_cellfinder_arguments(self) -> dict:
        """Passes voxel size data as one tuple instead of 3 individual floats"""
        data_input_dict = asdict(self)
        data_input_dict["voxel_sizes"] = (self.voxel_size_z, self.voxel_size_y, self.voxel_size_x)
        del data_input_dict["voxel_size_z"] # doesn't affect self, because asdict creates a copy.
        del data_input_dict["voxel_size_y"]
        del data_input_dict["voxel_size_x"]
        return data_input_dict

@dataclass
class DetectionInputs(InputContainer):
    """Container for cell candidate detection inputs."""
    soma_diameter: float
    ball_xy_size: float
    ball_z_size: float
    ball_overlap_fraction: float
    log_sigma_size: float
    n_sds_above_mean_thresh: int
    soma_spread_factor: float
    max_cluster_size: int

    persistent_defaults = {
        "soma_diameter" : 16.0,
        "ball_xy_size" :6,
        "ball_z_size" : 15,
        "ball_overlap" : 0.6,
        "filter_width" : 0.2,
        "threshold" : 10,
        "cell_spread" : 1.4,
        "max_cluster" : 100000
    }

@dataclass 
class ClassificationInputs(InputContainer):
    """Container for classification inputs."""
    trained_model : Path

    persistent_defaults = dict(trained_model=Path.home())

@dataclass 
class MiscInputs(InputContainer):
    """Container for miscellaneous inputs."""
    start_plane: int
    end_plane: int
    n_free_cpus: int
    analyse_local : bool
    debug : bool

    persistent_defaults = dict(
        start_plane=0,
        end_plane=0,
        number_of_free_cpus=2,
        analyse_local=False,
        debug=False,
    )

    def as_cellfinder_arguments(self) -> dict:
        misc_input_dict = asdict(self)
        del misc_input_dict["debug"]
        del misc_input_dict["analyse_local"]
        return misc_input_dict


def html_label_widget(label : str, tag : str = 'b'):
    return dict(
            widget_type="Label",
            label=f"<{tag}>{label}</{tag}>",
        )

def add_layers(points, viewer):
    """
    Adds classified cell candidates as two separate point layers to the napari viewer.
    """
    detected, rejected = cells_to_array(points)

    viewer.add_points(
        rejected,
        name="Rejected",
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightskyblue",
        visible=False,
        metadata=dict(point_type=Cell.UNKNOWN),
    )
    viewer.add_points(
        detected,
        name="Detected",
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightgoldenrodyellow",
        metadata=dict(point_type=Cell.CELL),
    )

@thread_worker
def run(
    data_inputs : DataInputs,
    detection_inputs: DetectionInputs,
    classification_inputs : ClassificationInputs,
    misc_inputs : MiscInputs,
    # Classification_batch_size,
):
    """Runs cellfinder in a separate thread, to prevent GUI blocking."""
    points = cellfinder_run(
        **data_inputs.as_cellfinder_arguments(),
        **detection_inputs.as_core_arguments(),
        **classification_inputs.as_core_arguments(),
        **misc_inputs.as_cellfinder_arguments(),
    )
    return points

