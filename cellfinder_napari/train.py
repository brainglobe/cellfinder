from pathlib import Path
from magicgui import magic_factory, widgets
from cellfinder_core.train.train_yml import run as train_yml
from cellfinder_core.download.models import model_weight_urls
from cellfinder_core.train.train_yml import models

MODELS = list(models.keys())
PRETRAINED_MODELS = list(model_weight_urls.keys())


def init(widget):
    widget.Learning_rate.native.setDecimals(4)
    widget.Learning_rate.value = 0.0001
    widget.insert(0, widgets.Label(value="<h2>cellfinder</h2>"))
    widget.insert(1, widgets.Label(value="<h3>Network training</h3>"))
    widget.insert(2, widgets.Label(value="<b>Training data:</b>"))
    widget.insert(5, widgets.Label(value="<b>Network (optional):</b>"))
    widget.insert(10, widgets.Label(value="<b>Training (optional):</b>"))
    widget.insert(21, widgets.Label(value="<b>Misc (optional):</b>"))


@magic_factory(
    YAML_files=dict(mode="rm", filter="*.yml"),
    Output_directory=dict(mode="d"),
    Test_fraction=dict(step=0.05, min=0.0, max=0.95),
    Learning_rate=dict(step=0.0001),
    Pretrained_model=dict(choices=PRETRAINED_MODELS),
    Model_depth=dict(choices=MODELS),
    call_button=True,
    widget_init=init,
)
def train(
    YAML_files: Path = Path.home(),
    Output_directory: Path = Path.home(),
    Trained_model: Path = Path.home(),
    Model_weights: Path = Path.home(),
    Model_depth: str = "50",
    Pretrained_model: str = PRETRAINED_MODELS[0],
    Continue_training: bool = False,
    Augment: bool = True,
    Tensorboard: bool = False,
    Save_weights: bool = False,
    Save_checkpoints: bool = True,
    Save_progress: bool = True,
    Epochs: int = 100,
    Learning_rate: float = 0.0001,
    Batch_size: int = 16,
    Test_fraction: float = 0.1,
    Number_of_free_cpus: int = 2,
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
    """
    if Trained_model == Path.home():
        Trained_model = None
    if Model_weights == Path.home():
        Model_weights = None

    if YAML_files[0] == Path.home():
        print("Please select a YAML file for training")
    else:
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
