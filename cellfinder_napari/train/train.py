from pathlib import Path
from typing import Optional

from cellfinder_core.download.models import model_weight_urls
from cellfinder_core.train.train_yml import models
from cellfinder_core.train.train_yml import run as train_yml
from magicgui import magicgui
from magicgui.widgets import FunctionGui, PushButton
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from cellfinder_napari.utils import (
    header_label_widget,
    html_label_widget,
    widget_header,
)

from .train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


@thread_worker
def run_training(
    training_data_inputs: TrainingDataInputs,
    optional_network_inputs: OptionalNetworkInputs,
    optional_training_inputs: OptionalTrainingInputs,
    misc_training_inputs: MiscTrainingInputs,
):
    print("Running training")
    train_yml(
        **training_data_inputs.as_core_arguments(),
        **optional_network_inputs.as_core_arguments(),
        **optional_training_inputs.as_core_arguments(),
        **misc_training_inputs.as_core_arguments(),
    )
    print("Finished!")


def training_widget() -> FunctionGui:
    @magicgui(
        header=header_label_widget,
        training_label=html_label_widget("Network training", tag="h3"),
        **TrainingDataInputs.widget_representation(),
        **OptionalNetworkInputs.widget_representation(),
        **OptionalTrainingInputs.widget_representation(),
        **MiscTrainingInputs.widget_representation(),
        call_button=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        header: dict,
        training_label: dict,
        data_options: dict,
        yaml_files: Path,
        output_directory: Path,
        network_options: dict,
        trained_model: Optional[Path],
        model_weights: Optional[Path],
        model_depth: str,
        pretrained_model: str,
        training_options: dict,
        continue_training: bool,
        augment: bool,
        tensorboard: bool,
        save_weights: bool,
        save_checkpoints: bool,
        save_progress: bool,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        test_fraction: float,
        misc_options: dict,
        number_of_free_cpus: int,
        reset_button: PushButton,
    ):
        """
        Parameters
        ----------
        yaml_files : Path
            YAML files containing paths to training data
        output_directory : Path
            Directory to save the output trained model
        trained_model : Optional[Path]
            Existing pre-trained model
        model_weights : Optional[Path]
            Existing pre-trained model weights
            Should be set along with "Model depth"
        model_depth : str
            ResNet model depth (as per He et al. (2015))
        pretrained_model : str
            Which pre-trained model to use
            (Supplied with cellfinder)
        continue_training : bool
            Continue training from an existing trained model
            If no trained model or model weights are specified,
            this will continue from the pretrained model
        augment : bool
            Augment the training data to improve generalisation
        tensorboard : bool
            Log to output_directory/tensorboard
        save_weights : bool
            Only store the model weights, and not the full model
            Useful to save storage space
        save_checkpoints : bool
            Store the model at intermediate points during training
        save_progress : bool
            Save training progress to a .csv file
        epochs : int
            Number of training epochs
            (How many times to use each training data point)
        learning_rate : float
            Learning rate for training the model
        batch_size : int
            Training batch size
        test_fraction : float
            Fraction of training data to use for validation
        number_of_free_cpus : int
            How many CPU cores to leave free
        reset_button : PushButton
            Reset parameters to default
        """
        trained_model = None if trained_model == Path.home() else trained_model
        model_weights = None if model_weights == Path.home() else model_weights

        training_data_inputs = TrainingDataInputs(yaml_files, output_directory)

        optional_network_inputs = OptionalNetworkInputs(
            trained_model,
            model_weights,
            model_depth,
            pretrained_model,
        )

        optional_training_inputs = OptionalTrainingInputs(
            continue_training,
            augment,
            tensorboard,
            save_weights,
            save_checkpoints,
            save_progress,
            epochs,
            learning_rate,
            batch_size,
            test_fraction,
        )

        misc_training_inputs = MiscTrainingInputs(number_of_free_cpus)

        if yaml_files[0] == Path.home():  # type: ignore
            show_info("Please select a YAML file for training")
        else:
            worker = run_training(
                training_data_inputs,
                optional_network_inputs,
                optional_training_inputs,
                misc_training_inputs,
            )
            worker.start()

    widget.header.value = widget_header
    widget.header.native.setOpenExternalLinks(True)

    @widget.reset_button.changed.connect
    def restore_defaults():
        defaults = {
            **TrainingDataInputs.defaults(),
            **OptionalNetworkInputs.defaults(),
            **OptionalTrainingInputs.defaults(),
            **MiscTrainingInputs.defaults(),
        }
        for name, value in defaults.items():
            # ignore fields with no default
            if value is not None:
                getattr(widget, name).value = value

    return widget
