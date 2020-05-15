import os
import sys
import platform
import pytest
import numpy as np
import pandas as pd

from brainio.brainio import load_nii
from imlib.general.string import get_text_lines

from cellfinder.main import main as cellfinder_run

data_dir = os.path.join(os.getcwd(), "tests", "data", "brain",)


test_output_dir = os.path.join(
    os.getcwd(), "tests", "data", "registration_output", platform.system()
)

x_pix = "40"
y_pix = "40"
z_pix = "50"

relative_tolerance = 0.01
absolute_tolerance = 10
check_less_precise_pd = 1


@pytest.mark.slow
def test_register(tmpdir, test_config_path):
    output_directory = os.path.join(str(tmpdir), "output")
    cellfinder_args = [
        "cellfinder",
        "-s",
        data_dir,
        "-b",
        data_dir,
        "-o",
        output_directory,
        "-x",
        x_pix,
        "-y",
        y_pix,
        "-z",
        z_pix,
        "--n-free-cpus",
        "0",
        "--registration-config",
        test_config_path,
        "--register",
        "--no-detection",
        "--no-classification",
    ]

    sys.argv = cellfinder_args
    cellfinder_run()
    output_directory = os.path.join(output_directory, "registration")

    # a hack because testing on linux on travis is 100% identical to local,
    # but windows is not
    if platform.system() == "Linux":
        image_list = [
            "annotations.nii",
            "boundaries.nii",
            "brain_filtered.nii",
            "control_point_file.nii",
            "downsampled.nii",
            "hemispheres.nii",
            "inverse_control_point_file.nii",
            "registered_atlas.nii",
            "registered_hemispheres.nii",
            "downsampled_channel_0.nii",
        ]
    else:
        image_list = [
            "annotations.nii",
            # "boundaries.nii",
            "brain_filtered.nii",
            "control_point_file.nii",
            "downsampled.nii",
            "hemispheres.nii",
            "inverse_control_point_file.nii",
            # "registered_atlas.nii",
            "registered_hemispheres.nii",
            "downsampled_channel_0.nii",
        ]

    for image in image_list:
        are_images_equal(image, output_directory, test_output_dir)

    assert get_text_lines(
        os.path.join(output_directory, "affine_matrix.txt")
    ) == get_text_lines(os.path.join(test_output_dir, "affine_matrix.txt"))

    assert get_text_lines(
        os.path.join(output_directory, "invert_affine_matrix.txt")
    ) == get_text_lines(
        os.path.join(test_output_dir, "invert_affine_matrix.txt")
    )

    pd.testing.assert_frame_equal(
        pd.read_csv(os.path.join(output_directory, "volumes.csv")),
        pd.read_csv(os.path.join(test_output_dir, "volumes.csv")),
        check_exact=False,
        check_less_precise=check_less_precise_pd,
    )


def are_images_equal(image_name, output_directory, test_output_directory):
    image = load_nii(
        os.path.join(output_directory, image_name),
        as_array=True,
        as_numpy=True,
    )
    test_image = load_nii(
        os.path.join(test_output_directory, image_name),
        as_array=True,
        as_numpy=True,
    )
    np.testing.assert_allclose(
        image, test_image, rtol=relative_tolerance, atol=absolute_tolerance
    )
