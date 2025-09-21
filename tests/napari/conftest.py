import pytest


@pytest.fixture
def create_napari_viewer(make_napari_viewer, caplog):
    """Shim for make_napari_viewer, that cleans up caplog from holding
    onto widgets. See issue #443.
    """
    yield make_napari_viewer

    # we only need to remove the logs if test didn't fail. Napari doesn't
    # check about memory leaks for failed tests
    for when in ("setup", "call", "teardown"):
        records = caplog.get_records(when)
        indices = [i for i, r in enumerate(records) if r.name == "in_n_out"]
        for i in indices[::-1]:
            del records[i]
