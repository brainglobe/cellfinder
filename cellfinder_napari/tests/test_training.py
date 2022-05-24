from cellfinder_napari.train import train


def test_add_training_widget(make_napari_viewer):
    """
    Smoke test to check that adding training widget works
    """
    viewer = make_napari_viewer()
    widget = train()
    viewer.window.add_dock_widget(widget)
