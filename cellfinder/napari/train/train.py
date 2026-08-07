from pathlib import Path
from typing import Literal, Optional

from magicgui import magicgui
from magicgui.widgets import FunctionGui, PushButton
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QScrollArea

from cellfinder.core.train.train_yaml import run as train_yaml
from cellfinder.napari.utils import cellfinder_header, html_label_widget

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
    show_info("Running training...")
    train_yaml(
        **training_data_inputs.as_core_arguments(),
        **optional_network_inputs.as_core_arguments(),
        **optional_training_inputs.as_core_arguments(),
        **misc_training_inputs.as_core_arguments(),
    )
    show_info("Training finished!")


def training_widget() -> FunctionGui:
    @magicgui(
        training_label=html_label_widget("Network training", tag="h3"),
        **TrainingDataInputs.widget_representation(),
        **OptionalNetworkInputs.widget_representation(),
        **OptionalTrainingInputs.widget_representation(),
        **MiscTrainingInputs.widget_representation(),
        call_button=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
        scrollable=True,
    )
    def widget(
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
        epochs: int,
        batch_size: int,
        test_fraction: float,
        normalize_channels: bool,
        learning_rate: float,
        lr_schedule: list[int],
        lr_multiplier: float,
        augment: bool,
        augment_likelihood: float,
        flippable_axis: list[Literal[0, 1, 2]],
        rotate_range: tuple[float, float],
        translate_range: tuple[float, float],
        scale_range: tuple[float, float],
        intensity_range: tuple[float, float],
        tensorboard: bool,
        save_checkpoints: bool,
        save_progress: bool,
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
        epochs : int
            Number of training epochs
            (How many times to use each training data point)
        batch_size : int
            Training batch size
        test_fraction : float
            Fraction of training data to use for validation
        normalize_channels : bool
            Whether to normalize the cubes by the mean/std of their origin
            dataset. If True, the yaml files must include the mean/std of
            the origin dataset.
        learning_rate : float
            Learning rate for training the model
        lr_schedule : list of ints
            If not empty, the list of epochs when to multiply the current
            learning rate by the lr_multiplier. E.g. if it's [10, 25], we start
            with a learning rate of 0.001, and `lr_multiplier` is 0.1, then the
            LR will be 0.001 for epochs 0-9, 0.0001 for 10-24, and 00001
            for epoch 25 and beyond.
        lr_multiplier : float
            The multiplier by which to multiply the previous learning rate
            at the epochs listed in `lr_schedule`.
        augment : bool
            Augment the training data to improve generalisation
        augment_likelihood: float
            A value in [0, 1] with the probability to augment with each
            transformation. Each transformation of each data sample is
            independently sampled with this probability.
        flippable_axis: list of axes indices
            A list of axes to potentially mirror around its center during
            augmentation. Values corresponds to the z, y, and x axes.
            E.g. if [0, 2] and `augment_likelihood` is 0.3, then axes 0 (z)
            and 2 (x) have each independently a 30% chance of being flipped
            around their centers.
        rotate_range: tuple of floats
            The interval we sample during augmentation for each data item to
            get a rotation value to rotate around the center of the sample in
            the x, y, and z axes (by the same value). If the range start and
            end is the same, no rotation is performed. Valid interval is
            `[-360, 360]` degrees.
        translate_range: tuple of floats
            The interval we sample during augmentation for each data item to
            get a translation value to translate the sample along the x, y,
            and z axes (by the same value). If the range start and end is the
            same, no translation is performed. Meaningful interval is
            `[-1, 1]`, where `0` means none and `1` means translate by a full
            sample size positively.
        scale_range: tuple of floats
            The interval we sample during augmentation for each data item to
            get a scaling value to scale the sample from its center in
            the x, y, and z axes (by the same value). If the scale start and
            end is the same, no scaling is performed. Valid interval is
            `(0, inf]`, where `1` is the original scale.
        intensity_range: tuple of floats
            The interval we sample during augmentation for each data item to
            get a multiplication factor to multiply the sample voxels. If the
            range start and end is the same, no multiplication is performed.
            Valid interval is `(0, inf]`, where `1` is the original intensity.
        tensorboard : bool
            Log to output_directory/tensorboard
        save_checkpoints : bool
            Store the model at intermediate points during training
        save_progress : bool
            Save training progress to a .csv file
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
            save_checkpoints,
            save_progress,
            epochs,
            learning_rate,
            batch_size,
            test_fraction,
            lr_schedule,
            lr_multiplier,
            normalize_channels,
            augment_likelihood,
            flippable_axis,
            rotate_range,
            translate_range,
            scale_range,
            intensity_range,
        )

        misc_training_inputs = MiscTrainingInputs(number_of_free_cpus)

        if yaml_files[0] == Path.home():  # type: ignore
            show_info("Please select a YAML file for training")
        else:
            show_info("Starting training process...")
            worker = run_training(
                training_data_inputs,
                optional_network_inputs,
                optional_training_inputs,
                misc_training_inputs,
            )
            worker.start()

    widget.native.layout().insertWidget(0, cellfinder_header())

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

    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget
