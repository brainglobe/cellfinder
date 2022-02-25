import napari
from abc import abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path

from cellfinder_napari.utils import html_label_widget


class InputContainer:
    """Base for classes that contain inputs

    Intended to be derived to group specific related widget inputs (e.g from the same widget section)
    into a container. Derived classes should be Python data classes.

    Enforces common interfaces for
    - how to get default values for the inputs
    - how inputs are passed to cellfinder core
    - how the inputs are shown in the widget
    """

    @classmethod
    def defaults(cls) -> dict:
        """Returns default values of this class's fields as a dict."""
        # Derived classes are not expected to be particularly
        # slow to instantiate, so use the default constructor
        # to avoid code repetition.
        return asdict(cls())

    @abstractmethod
    def as_core_arguments(self) -> dict:
        """Determines how dataclass fields are passed to cellfinder-core.

        The implementation provided here can be re-used in derived classes, if convenient.
        """
        # note that asdict returns a new instance of a dict,
        # so any subsequent modifications of this dict won't affect the class instance
        return asdict(self)

    @classmethod
    def _numerical_widget(
        cls, key : str, custom_label : str = None, step=None, min=None, max=None
    ) -> dict:
        """Represents a numerical field, given by key, as a formatted widget with the field's default value.

        The widget's label, step, min, and max can be adjusted.
        """
        if custom_label is None:
            label = key.replace("_", " ").capitalize()
        else:
            label = custom_label
        value = cls.defaults()[key]
        if step is None:
             step=1 if type(step)==int else 0.1
        if min is None and max is None:
            return dict(value=value, label=label, step=step)
        else:
            return dict(
                value=value,
                label=label,
                step=step,
                min=min,
                max=max,
            )

    @classmethod
    @abstractmethod
    def widget_representation(cls) -> dict:
        """What the class will look like as a napari widget"""
        pass


@dataclass
class DataInputs(InputContainer):
    """Container for image-related ("Data") inputs."""

    signal_array: napari.layers.Image = None
    background_array: napari.layers.Image = None
    voxel_size_z: float = 5
    voxel_size_y: float = 5
    voxel_size_x: float = 2

    def as_core_arguments(self) -> dict:
        """Passes voxel size data as one tuple instead of 3 individual floats"""
        data_input_dict = super().as_core_arguments()
        data_input_dict["voxel_sizes"] = (
            self.voxel_size_z,
            self.voxel_size_y,
            self.voxel_size_x,
        )
        del data_input_dict["voxel_size_z"]
        del data_input_dict["voxel_size_y"]
        del data_input_dict["voxel_size_x"]
        return data_input_dict

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            data_options=html_label_widget("Data:"),
            voxel_size_z=cls._numerical_widget(
                "voxel_size_z", custom_label="Voxel size (z)"
            ),
            voxel_size_y=cls._numerical_widget(
                "voxel_size_y", custom_label="Voxel size (y)"
            ),
            voxel_size_x=cls._numerical_widget(
                "voxel_size_x", custom_label="Voxel size (x)"
            ),
        )


@dataclass
class DetectionInputs(InputContainer):
    """Container for cell candidate detection inputs."""

    soma_diameter: float = 16.0
    ball_xy_size: float = 6
    ball_z_size: float = 15
    ball_overlap_fraction: float = 0.6
    log_sigma_size: float = 0.2
    n_sds_above_mean_thresh: int = 10
    soma_spread_factor: float = 1.4
    max_cluster_size: int = 100000

    def as_core_arguments(self) -> dict:
        return super().as_core_arguments()

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            detection_options=html_label_widget("Detection:"),
            soma_diameter=cls._numerical_widget("soma_diameter"),
            ball_xy_size=cls._numerical_widget(
                "ball_xy_size", custom_label="Ball filter (xy)"
            ),
            ball_z_size=cls._numerical_widget(
                "ball_z_size", custom_label="Ball filter (z)"
            ),
            ball_overlap_fraction=cls._numerical_widget(
                "ball_overlap_fraction", custom_label="Ball overlap"
            ),
            log_sigma_size=cls._numerical_widget(
                "log_sigma_size", custom_label="Filter width"
            ),
            n_sds_above_mean_thresh=cls._numerical_widget(
                "n_sds_above_mean_thresh", custom_label="Threshold"
            ),
            soma_spread_factor=cls._numerical_widget(
                "soma_spread_factor", custom_label="Cell spread"
            ),
            max_cluster_size=cls._numerical_widget(
                "max_cluster_size",
                custom_label="Max cluster",
                min=0,
                max=10000000,
            ),
        )


@dataclass
class ClassificationInputs(InputContainer):
    """Container for classification inputs."""

    trained_model: Path = Path.home()

    def as_core_arguments(self) -> dict:
        return super().as_core_arguments()

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            classification_options=html_label_widget("Classification:"),
            trained_model=dict(value=cls.defaults()["trained_model"]),
        )


@dataclass
class MiscInputs(InputContainer):
    """Container for miscellaneous inputs."""

    start_plane: int = 0
    end_plane: int = 0
    n_free_cpus: int = 2
    analyse_local: bool = False
    debug: bool = False

    def as_core_arguments(self) -> dict:
        misc_input_dict = super().as_core_arguments()
        del misc_input_dict["debug"]
        del misc_input_dict["analyse_local"]
        return misc_input_dict

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            misc_options=html_label_widget("Miscellaneous:"),
            start_plane=cls._numerical_widget("start_plane", min=0, max=100000),
            end_plane=cls._numerical_widget("end_plane", min=0, max=100000),
            n_free_cpus=cls._numerical_widget(
                "n_free_cpus", custom_label="Number of free CPUs"
            ),
            analyse_local=dict(value=cls.defaults()["analyse_local"]),
            debug=dict(value=cls.defaults()["debug"]),
        )
