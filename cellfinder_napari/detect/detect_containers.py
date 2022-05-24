from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import napari
import numpy

from cellfinder_napari.input_container import InputContainer
from cellfinder_napari.utils import html_label_widget


@dataclass
class DataInputs(InputContainer):
    """Container for image-related ("Data") inputs."""

    signal_array: numpy.ndarray = None
    background_array: numpy.ndarray = None
    voxel_size_z: float = 5
    voxel_size_y: float = 2
    voxel_size_x: float = 2

    def as_core_arguments(self) -> dict:
        """Passes voxel size data as one tuple instead of 3 individual floats"""
        data_input_dict = super().as_core_arguments()
        data_input_dict["voxel_sizes"] = (
            self.voxel_size_z,
            self.voxel_size_y,
            self.voxel_size_x,
        )
        # del operator doesn't affect self, because asdict creates a copy of fields.
        del data_input_dict["voxel_size_z"]
        del data_input_dict["voxel_size_y"]
        del data_input_dict["voxel_size_x"]
        return data_input_dict

    @property
    def nplanes(self):
        return len(self.signal_array)

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            data_options=html_label_widget("Data:"),
            voxel_size_z=cls._custom_widget(
                "voxel_size_z", custom_label="Voxel size (z)"
            ),
            voxel_size_y=cls._custom_widget(
                "voxel_size_y", custom_label="Voxel size (y)"
            ),
            voxel_size_x=cls._custom_widget(
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
            soma_diameter=cls._custom_widget("soma_diameter"),
            ball_xy_size=cls._custom_widget(
                "ball_xy_size", custom_label="Ball filter (xy)"
            ),
            ball_z_size=cls._custom_widget(
                "ball_z_size", custom_label="Ball filter (z)"
            ),
            ball_overlap_fraction=cls._custom_widget(
                "ball_overlap_fraction", custom_label="Ball overlap"
            ),
            log_sigma_size=cls._custom_widget(
                "log_sigma_size", custom_label="Filter width"
            ),
            n_sds_above_mean_thresh=cls._custom_widget(
                "n_sds_above_mean_thresh", custom_label="Threshold"
            ),
            soma_spread_factor=cls._custom_widget(
                "soma_spread_factor", custom_label="Cell spread"
            ),
            max_cluster_size=cls._custom_widget(
                "max_cluster_size",
                custom_label="Max cluster",
                min=0,
                max=10000000,
            ),
        )


@dataclass
class ClassificationInputs(InputContainer):
    """Container for classification inputs."""

    use_pre_trained_weights: bool = True
    trained_model: Optional[Path] = Path.home()

    def as_core_arguments(self) -> dict:
        args = super().as_core_arguments()
        del args["use_pre_trained_weights"]
        return args

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            classification_options=html_label_widget("Classification:"),
            use_pre_trained_weights=dict(
                value=cls.defaults()["use_pre_trained_weights"]
            ),
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
            start_plane=cls._custom_widget("start_plane", min=0, max=100000),
            end_plane=cls._custom_widget("end_plane", min=0, max=100000),
            n_free_cpus=cls._custom_widget(
                "n_free_cpus", custom_label="Number of free CPUs"
            ),
            analyse_local=dict(value=cls.defaults()["analyse_local"]),
            debug=dict(value=cls.defaults()["debug"]),
        )
