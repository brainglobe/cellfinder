from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cellfinder_core.download.models import model_weight_urls
from cellfinder_core.train.train_yml import models
from magicgui.types import FileDialogMode

from cellfinder_napari.input_container import InputContainer
from cellfinder_napari.utils import html_label_widget


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
                filter="*.y?ml",
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
    pretrained_model: str = str(list(model_weight_urls.keys())[0])

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
                choices=list(model_weight_urls.keys()),
            ),
        )


@dataclass
class OptionalTrainingInputs(InputContainer):
    continue_training: bool = False
    augment: bool = True
    tensorboard: bool = False
    save_weights: bool = False
    save_checkpoints: bool = True
    save_progress: bool = True
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 16
    test_fraction: float = 0.1

    def as_core_arguments(self) -> dict:
        arguments = super().as_core_arguments()
        arguments["no_augment"] = not arguments.pop("augment")
        arguments["no_save_checkpoints"] = not arguments.pop(
            "save_checkpoints"
        )
        return arguments

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            training_options=html_label_widget("Training (optional)"),
            continue_training=cls._custom_widget("continue_training"),
            augment=cls._custom_widget("augment"),
            tensorboard=cls._custom_widget("tensorboard"),
            save_weights=cls._custom_widget("save_weights"),
            save_checkpoints=cls._custom_widget("save_checkpoints"),
            save_progress=cls._custom_widget("save_progress"),
            epochs=cls._custom_widget("epochs"),
            learning_rate=cls._custom_widget("learning_rate", step=1e-4),
            batch_size=cls._custom_widget("batch_size"),
            test_fraction=cls._custom_widget(
                "test_fraction", step=0.05, min=0.05, max=0.95
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
