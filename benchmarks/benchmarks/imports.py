# ------------------------------------
# Runtime benchmarks
# ------------------------------------
def timeraw_import_main():
    return """
    from cellfinder_core.main import main
    """


def timeraw_import_io_dask():
    return """
    from cellfinder_core.tools.IO import read_with_dask
    """


def timeraw_import_io_tiff_meta():
    return """
    from cellfinder_core.tools.IO import get_tiff_meta
    """


def timeraw_import_prep_tensorflow():
    return """
    from cellfinder_core.tools.prep import prep_tensorflow
    """


def timeraw_import_prep_models():
    return """
    from cellfinder_core.tools.prep import prep_models
    """


def timeraw_import_prep_classification():
    return """
    from cellfinder_core.tools.prep import prep_classification
    """


def timeraw_import_prep_training():
    return """
    from cellfinder_core.tools.prep import prep_training
    """
