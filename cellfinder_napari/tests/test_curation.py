from cellfinder_napari import sample_data
from cellfinder_napari.curation import CurationWidget


def test_detect_function(make_napari_viewer):
    """
    Smoke test that adding the curation widget to napari works
    """
    viewer = make_napari_viewer()
    widget = CurationWidget(viewer)
    viewer.window.add_dock_widget(widget)
