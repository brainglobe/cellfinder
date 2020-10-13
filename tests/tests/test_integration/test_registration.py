import os
import sys
import platform
import pytest
import numpy as np
import pandas as pd

from imio.load import load_any

from cellfinder.main import main as cellfinder_run

data_dir = os.path.join(
    os.getcwd(),
    "tests",
    "data",
    "brain",
)

test_niftyreg_output = os.path.join(
    os.getcwd(), "tests", "data", "registration_output", platform.system()
)

x_pix = "40"
y_pix = "40"
z_pix = "50"

relative_tolerance = 0.01
absolute_tolerance = 10
check_less_precise_pd = 1


@pytest.mark.xfail(reason="Issues across machines")
@pytest.mark.slow
def test_registration_niftyreg(tmpdir):
    output_directory = os.path.join(str(tmpdir), "output")
    cellfinder_args = [
        "cellfinder",
        "-s",
        data_dir,
        "-b",
        data_dir,
        "-o",
        output_directory,
        "-v",
        z_pix,
        y_pix,
        x_pix,
        "--orientation",
        "psl",
        "--n-free-cpus",
        "0",
        "--atlas",
        "allen_mouse_100um",
        "--no-detection",
        "--no-classification",
        "--no-analyse",
        "--no-figures",
    ]

    sys.argv = cellfinder_args
    cellfinder_run()

    # none of this testing is ideal, as results seem to vary between systems

    if platform.system() == "Linux":
        image_list = [
            "boundaries.tiff",
            "deformation_field_0.tiff",
            "deformation_field_1.tiff",
            "deformation_field_2.tiff",
            "downsampled.tiff",
            "downsampled_channel_0.tiff",
            "downsampled_standard.tiff",
            "downsampled_standard_channel_0.tiff",
            "registered_atlas.tiff",
            "registered_hemispheres.tiff",
        ]
    else:
        image_list = [
            "boundaries.tiff",
            "deformation_field_0.tiff",
            "deformation_field_1.tiff",
            "deformation_field_2.tiff",
            "downsampled.tiff",
            "downsampled_channel_0.tiff",
            # "downsampled_standard.tiff",
            # "downsampled_standard_channel_0.tiff",
            # "registered_atlas.tiff",
            # "registered_hemispheres.tiff",
        ]
    output_directory = os.path.join(output_directory, "registration")
    for image in image_list:
        are_images_equal(image, output_directory, test_niftyreg_output)

    if platform.system() == "Linux":
        pd.testing.assert_frame_equal(
            pd.read_csv(os.path.join(output_directory, "volumes.csv")),
            pd.read_csv(os.path.join(test_niftyreg_output, "volumes.csv")),
        )


def are_images_equal(image_name, output_directory, test_output_directory):
    image = load_any(
        os.path.join(output_directory, image_name),
    )
    test_image = load_any(
        os.path.join(test_output_directory, image_name),
    )
    np.testing.assert_allclose(
        image, test_image, rtol=relative_tolerance, atol=absolute_tolerance
    )
