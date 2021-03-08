from pathlib import Path
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from cellfinder_core.train.train_yml import run as run_training

# TODO:
# how to store & fetch pre-trained models?
# how to support N yaml files

"""A LineEdit widget with a button that opens a FileDialog.

Parameters
----------
mode : FileDialogMode or str
    - ``'r'`` returns one existing file.
    - ``'rm'`` return one or more existing files.
    - ``'w'`` return one file name that does not have to exist.
    - ``'d'`` returns one existing directory.
filter : str, optional
    The filter is used to specify the kind of files that should be shown.
    It should be a glob-style string, like ``'*.png'`` (this may be
    backend-specific)
"""


@magic_factory(
    YAML_file={"mode": "d"},
    call_button=True,
)
def train(
    YAML_file: Path = Path.home(),
    Output_directory: Path = Path.home(),
    Epochs: int = 100,
    Learning_rate: float = 0.0001,
    Batch_size: int = 16,
    Number_of_free_cpus: int = 2,
):

    run_training(
        Output_directory,
        YAML_file,
        n_free_cpus=Number_of_free_cpus,
        epochs=Epochs,
        learning_rate=Learning_rate,
        batch_size=Batch_size,
    )


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return train, {"name": "Train network"}
