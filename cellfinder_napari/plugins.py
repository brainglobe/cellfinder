from cellfinder_napari.curation import CurationWidget
from cellfinder_napari.detect import detect
from cellfinder_napari.train import train


def napari_experimental_provide_dock_widget():
    return [
        (detect, {"name": "Cell detection"}),
        (train, {"name": "Train network"}),
        (CurationWidget, {"name": "Curation"}),
    ]
