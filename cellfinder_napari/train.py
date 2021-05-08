from pathlib import Path
from magicgui import magicgui
from napari.qt.threading import thread_worker
from cellfinder_napari.utils import brainglobe_logo


def train():
    from cellfinder_core.download.models import model_weight_urls
    from cellfinder_core.train.train_yml import models

    MODELS = list(models.keys())
    PRETRAINED_MODELS = list(model_weight_urls.keys())
    DEFAULT_PARAMETERS = dict(
        YAML_files=Path.home(),
        Output_directory=Path.home(),
        Trained_model=Path.home(),
        Model_weights=Path.home(),
        Model_depth="50",
        Pretrained_model=PRETRAINED_MODELS[0],
        Continue_training=False,
        Augment=True,
        Tensorboard=False,
        Save_weights=False,
        Save_checkpoints=True,
        Save_progress=True,
        Epochs=100,
        Learning_rate=0.0001,
        Batch_size=16,
        Test_fraction=0.1,
        Number_of_free_cpus=2,
    )

    @magicgui(
        header=dict(
            widget_type="Label",
            label=f'<h1><img src="{brainglobe_logo}"width="100">cellfinder</h1>',
        ),
        training_label=dict(
            widget_type="Label",
            label="<h3>Network training</h3>",
        ),
        data_options=dict(
            widget_type="Label",
            label="<b>Training Data:</b>",
        ),
        network_options=dict(
            widget_type="Label",
            label="<b>Network (optional):</b>",
        ),
        training_options=dict(
            widget_type="Label",
            label="<b>Training (optional):</b>",
        ),
        misc_options=dict(
            widget_type="Label",
            label="<b>Misc (optional):</b>",
        ),
        YAML_files=dict(
            value=DEFAULT_PARAMETERS["YAML_files"], mode="rm", filter="*.yml"
        ),
        Output_directory=dict(
            value=DEFAULT_PARAMETERS["Output_directory"], mode="d"
        ),
        Trained_model=dict(value=DEFAULT_PARAMETERS["Trained_model"]),
        Model_weights=dict(value=DEFAULT_PARAMETERS["Model_weights"]),
        Model_depth=dict(
            value=DEFAULT_PARAMETERS["Model_depth"], choices=MODELS
        ),
        Pretrained_model=dict(
            value=DEFAULT_PARAMETERS["Pretrained_model"],
            choices=PRETRAINED_MODELS,
        ),
        Continue_training=dict(
            value=DEFAULT_PARAMETERS["Continue_training"],
            label="Continue training",
        ),
        Augment=dict(value=DEFAULT_PARAMETERS["Augment"]),
        Tensorboard=dict(value=DEFAULT_PARAMETERS["Tensorboard"]),
        Save_weights=dict(
            value=DEFAULT_PARAMETERS["Save_weights"], label="Save weights"
        ),
        Save_checkpoints=dict(
            value=DEFAULT_PARAMETERS["Save_checkpoints"],
            label="Save checkpoints",
        ),
        Save_progress=dict(
            value=DEFAULT_PARAMETERS["Save_progress"], label="Save progress"
        ),
        Epochs=dict(value=DEFAULT_PARAMETERS["Epochs"]),
        Learning_rate=dict(
            value=DEFAULT_PARAMETERS["Learning_rate"], step=0.0001
        ),
        Batch_size=dict(value=DEFAULT_PARAMETERS["Batch_size"]),
        Test_fraction=dict(
            value=DEFAULT_PARAMETERS["Test_fraction"],
            step=0.05,
            min=0.0,
            max=0.95,
        ),
        Number_of_free_cpus=dict(
            value=DEFAULT_PARAMETERS["Number_of_free_cpus"]
        ),
        call_button=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        header,
        training_label,
        data_options,
        YAML_files: Path,
        Output_directory: Path,
        network_options,
        Trained_model: Path,
        Model_weights: Path,
        Model_depth: str,
        Pretrained_model: str,
        training_options,
        Continue_training: bool,
        Augment: bool,
        Tensorboard: bool,
        Save_weights: bool,
        Save_checkpoints: bool,
        Save_progress: bool,
        Epochs: int,
        Learning_rate: float,
        Batch_size: int,
        Test_fraction: float,
        misc_options,
        Number_of_free_cpus: int,
        reset_button,
    ):
        """

        Parameters
        ----------
        YAML_files : Path
            YAML files containing paths to training data
        Output_directory : Path
            Directory to save the output trained model
        Trained_model : Path
            Existing pre-trained model
        Model_weights : Path
            Existing pre-trained model weights
            Should be set along with "Model depth"
        Model_depth : str
            ResNet model depth (as per He et al. (2015)
        Pretrained_model : str
            Which pre-trained model to use
            (Supplied with cellfinder)
        Continue_training : bool
            Continue training from an existing trained model
            If no trained model or model weights are specified,
            this will continue from the pretrained model
        Augment : bool
            Augment the training data to improve generalisation
        Tensorboard : bool
            Log to output_directory/tensorboard
        Save_weights : bool
            Only store the model weights, and not the full model
            Useful to save storage space
        Save_checkpoints : bool
            Store the model at intermediate points during training
        Save_progress : bool
            Save training progress to a .csv file
        Epochs : int
            Number of training epochs
            (How many times to use each training data point)
        Learning_rate : float
            Learning rate for training the model
        Batch_size : int
            Training batch size
        Test_fraction : float
            Fraction of training data to use for validation
        Number_of_free_cpus : int
            How many CPU cores to leave free
        reset_button :
            Reset parameters to default
        """
        if Trained_model == Path.home():
            Trained_model = None
        if Model_weights == Path.home():
            Model_weights = None

        if YAML_files[0] == Path.home():
            print("Please select a YAML file for training")
        else:
            worker = run_training(
                Output_directory,
                YAML_files,
                Model_depth,
                Pretrained_model,
                Trained_model,
                Model_weights,
                Augment,
                Tensorboard,
                Save_weights,
                Save_progress,
                Save_checkpoints,
                Number_of_free_cpus,
                Continue_training,
                Test_fraction,
                Epochs,
                Learning_rate,
                Batch_size,
            )
            worker.start()

    widget.header.value = (
        "<p>Efficient cell detection in large images.</p>"
        '<p><a href="https://cellfinder.info" style="color:gray;">Website</a></p>'
        '<p><a href="https://docs.brainglobe.info/cellfinder/napari-plugin" style="color:gray;">Documentation</a></p>'
        '<p><a href="https://github.com/brainglobe/cellfinder-napari" style="color:gray;">Source</a></p>'
        '<p><a href="https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2" style="color:gray;">Citation</a></p>'
        "<p><small>For help, hover the cursor over each parameter.</small>"
    )
    widget.header.native.setOpenExternalLinks(True)

    @widget.reset_button.changed.connect
    def restore_defaults(event=None):
        for name, value in DEFAULT_PARAMETERS.items():
            getattr(widget, name).value = value

    return widget


@thread_worker
def run_training(
    Output_directory,
    YAML_files,
    Model_depth,
    Pretrained_model,
    Trained_model,
    Model_weights,
    Augment,
    Tensorboard,
    Save_weights,
    Save_progress,
    Save_checkpoints,
    Number_of_free_cpus,
    Continue_training,
    Test_fraction,
    Epochs,
    Learning_rate,
    Batch_size,
):
    from cellfinder_core.train.train_yml import run as train_yml

    print("Running training")
    train_yml(
        Output_directory,
        YAML_files,
        network_depth=Model_depth,
        model=Pretrained_model,
        trained_model=Trained_model,
        model_weights=Model_weights,
        no_augment=not Augment,
        tensorboard=Tensorboard,
        save_weights=Save_weights,
        save_progress=Save_progress,
        no_save_checkpoints=not Save_checkpoints,
        n_free_cpus=Number_of_free_cpus,
        continue_training=Continue_training,
        test_fraction=Test_fraction,
        epochs=Epochs,
        learning_rate=Learning_rate,
        batch_size=Batch_size,
    )
    print("Finished!")
