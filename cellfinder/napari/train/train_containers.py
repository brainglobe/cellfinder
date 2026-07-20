from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from magicgui.types import FileDialogMode

from cellfinder.core.classify.augment import (
    interval_to_rand_range,
    interval_to_rand_range_3d,
)
from cellfinder.core.download.download import model_filenames
from cellfinder.core.train.train_yaml import models
from cellfinder.napari.input_container import InputContainer
from cellfinder.napari.utils import html_label_widget


@dataclass
class TrainingDataInputs(InputContainer):
    """Container for Training Data input widgets"""

    yaml_files: Path = Path.home()
    output_directory: Path = Path.home()

    def as_core_arguments(self) -> dict:
        arguments = super().as_core_arguments()
        arguments["output_dir"] = arguments.pop("output_directory")
        arguments["yaml_file"] = arguments.pop("yaml_files")
        return arguments

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            data_options=html_label_widget("Training Data:"),
            yaml_files=cls._custom_widget(
                "yaml_files",
                custom_label="YAML files",
                mode=FileDialogMode.EXISTING_FILES,
                filter="*.yaml",
            ),
            output_directory=cls._custom_widget(
                "output_directory", mode=FileDialogMode.EXISTING_DIRECTORY
            ),
        )


@dataclass
class OptionalNetworkInputs(InputContainer):
    """Container for Optional Network input widgets"""

    trained_model: Optional[Path] = Path.home()
    model_weights: Optional[Path] = Path.home()
    model_depth: str = list(models.keys())[2]
    pretrained_model: str = str(list(model_filenames.keys())[0])

    def as_core_arguments(self) -> dict:
        arguments = super().as_core_arguments()
        arguments["model"] = arguments.pop("pretrained_model")
        arguments["network_depth"] = arguments.pop("model_depth")
        return arguments

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            network_options=html_label_widget("Network (optional)"),
            trained_model=cls._custom_widget("trained_model"),
            model_weights=cls._custom_widget("model_weights"),
            model_depth=cls._custom_widget(
                "model_depth", choices=list(models.keys())
            ),
            pretrained_model=cls._custom_widget(
                "pretrained_model",
                choices=list(model_filenames.keys()),
            ),
        )


@dataclass
class OptionalTrainingInputs(InputContainer):
    continue_training: bool = False
    augment: bool = True
    tensorboard: bool = False
    save_checkpoints: bool = True
    save_progress: bool = True
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 16
    test_fraction: float = 0.1
    lr_schedule: list[int] | tuple[int, ...] = ()
    lr_multiplier: float = 0.1
    normalize_channels: bool = False
    augment_likelihood: float = 0.9
    flippable_axis: list[int] = field(default_factory=lambda: [0, 1, 2])
    rotate_range: tuple[float, float] = -1, 1
    translate_range: tuple[float, float] = -0.05, 0.05
    scale_range: tuple[float, float] = 1, 1
    intensity_range: tuple[float, float] = 1, 1

    def as_core_arguments(self) -> dict:
        arguments = super().as_core_arguments()
        arguments["no_augment"] = not arguments.pop("augment")
        arguments["no_save_checkpoints"] = not arguments.pop(
            "save_checkpoints"
        )
        arguments["flippable_axis"] = list(set(arguments["flippable_axis"]))
        for arg in ("rotate_range", "translate_range", "scale_range"):
            arguments[arg] = interval_to_rand_range_3d(*arguments[arg])
        arguments["intensity_range"] = interval_to_rand_range(
            *arguments["intensity_range"]
        )

        return arguments

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            training_options=html_label_widget("Training options"),
            continue_training=cls._custom_widget("continue_training"),
            augment=cls._custom_widget("augment"),
            tensorboard=cls._custom_widget("tensorboard"),
            save_checkpoints=cls._custom_widget("save_checkpoints"),
            save_progress=cls._custom_widget("save_progress"),
            epochs=cls._custom_widget("epochs"),
            learning_rate=cls._custom_widget("learning_rate", step=1e-4),
            batch_size=cls._custom_widget("batch_size"),
            test_fraction=cls._custom_widget(
                "test_fraction", step=0.05, min=0.05, max=0.95
            ),
            lr_schedule=cls._custom_widget(
                "lr_schedule", custom_label="LR schedule"
            ),
            lr_multiplier=cls._custom_widget(
                "lr_multiplier", custom_label="LR multiplier"
            ),
            normalize_channels=cls._custom_widget("normalize_channels"),
            augment_likelihood=cls._custom_widget(
                "augment_likelihood", step=0.01, min=0.00, max=1
            ),
            flippable_axis=cls._custom_widget("flippable_axis"),
            rotate_range=cls._custom_widget(
                "rotate_range",
                options={"step": 1, "min": -360, "max": 360},
            ),
            translate_range=cls._custom_widget(
                "translate_range",
                options={"step": 0.01, "min": -1, "max": 1},
            ),
            scale_range=cls._custom_widget(
                "scale_range",
                options={"step": 0.01, "min": 0.00, "max": 100},
            ),
            intensity_range=cls._custom_widget(
                "intensity_range",
                options={"step": 0.01, "min": 0.00, "max": 1000},
            ),
        )


@dataclass
class MiscTrainingInputs(InputContainer):
    number_of_free_cpus: int = 2

    def as_core_arguments(self) -> dict:
        return dict(n_free_cpus=self.number_of_free_cpus)

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            misc_options=html_label_widget("Misc (optional):"),
            number_of_free_cpus=cls._custom_widget(
                "number_of_free_cpus", custom_label="Number of free CPUs"
            ),
        )
