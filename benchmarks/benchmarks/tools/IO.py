from pathlib import Path

from cellfinder_core.tools.IO import get_tiff_meta, read_with_dask

CELLFINDER_CORE_PATH = Path(__file__).parents[3]
TESTS_DATA_INTEGRATION_PATH = (
    Path(CELLFINDER_CORE_PATH) / "tests" / "data" / "integration"
)


class Read:
    # ------------------------------------
    # Data
    # ------------------------------
    detection_crop_planes_ch0 = TESTS_DATA_INTEGRATION_PATH / Path(
        "detection", "crop_planes", "ch0"
    )
    detection_crop_planes_ch1 = TESTS_DATA_INTEGRATION_PATH / Path(
        "detection", "crop_planes", "ch1"
    )
    cells_tif_files = list(
        Path(TESTS_DATA_INTEGRATION_PATH, "training", "cells").glob("*.tif")
    )
    non_cells_tif_files = list(
        Path(TESTS_DATA_INTEGRATION_PATH, "training", "non_cells").glob(
            "*.tif"
        )
    )

    # ---------------------------------------------
    # Setup function
    # --------------------------------------------
    def setup(self, subdir):
        self.data_dir = str(subdir)

    # ---------------------------------------------
    # Reading 3d arrays with dask
    # --------------------------------------------
    def time_read_with_dask(self, subdir):
        read_with_dask(self.data_dir)

    # parameters to sweep across
    time_read_with_dask.param_names = [
        "tests_data_integration_subdir",
    ]
    time_read_with_dask.params = (
        [detection_crop_planes_ch0, detection_crop_planes_ch1],
    )

    # -----------------------------------------------
    # Reading metadata from tif files
    # -------------------------------------------------
    def time_get_tiff_meta(
        self,
        subdir,
    ):
        get_tiff_meta(self.data_dir)

    # parameters to sweep across
    time_get_tiff_meta.param_names = [
        "tests_data_integration_tiffile",
    ]

    time_get_tiff_meta.params = cells_tif_files + non_cells_tif_files
