import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from skimage import segmentation as sk_segmentation

from brainio import brainio
from imlib.image.scale import scale_and_convert_to_16_bits
from imlib.image import nii

from cellfinder.tools.source_files import source_custom_config
from cellfinder.tools.source_files import get_structures_path
from cellfinder.analyse.group.region_summary import get_substructures


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--structure-names",
        dest="structure_names",
        type=str,
        nargs="+",
        required=True,
        help="Structure name as a string (as per the reference atlas)",
    )

    parser.add_argument(
        "-a",
        "--atlas",
        dest="atlas_path",
        type=str,
        help="Path to the atlas image that the resulting "
        "image will be based on",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        type=str,
        help="Output .nii filename.",
    )
    parser.add_argument(
        "--glass",
        dest="glass",
        action="store_true",
        help="Generate a hollow volume",
    )
    parser.add_argument(
        "--atlas-config",
        dest="atlas_config",
        type=str,
        help="To supply your own, custom atlas configuration file.",
    )
    return parser


def generate_region_volume(
    structure_names, atlas_path, output_path, atlas_config, glass=False
):
    structure_csv_file = get_structures_path()
    reference_structures_table = pd.read_csv(structure_csv_file)

    # ensure all names are valid
    for indv_structure_name in structure_names:
        try:
            get_substructures(indv_structure_name, reference_structures_table)
        except IndexError:
            raise ValueError(
                f"Brain region: '{indv_structure_name}' cannot be found "
                f"in file: {structure_csv_file}. Please choose "
                f"another structure."
            )

    print(f"Loading atlas from: {atlas_path}")
    atlas = brainio.load_nii(atlas_path, as_array=False)
    atlas_scale = atlas.header.get_zooms()
    atlas = atlas.get_data()

    transformation_matrix = nii.get_transformation_matrix(atlas_config)

    if len(structure_names) > 1:
        # Initialise an image to add each subimage to.
        final_image = np.zeros_like(atlas)

    for indv_structure_name in structure_names:
        print(f"Analysing brain region: {indv_structure_name}")
        substructures = get_substructures(
            indv_structure_name, reference_structures_table
        )

        print("This includes structures:")
        indv_substructure_names = substructures["name"].values
        for indv_substructure_name in indv_substructure_names:
            print(indv_substructure_name)

        list_vals = substructures["id"].values

        print("Generating image with specified regions \n")
        sub_image = np.isin(atlas, list_vals)

        if glass:
            print("Generating glass brain")
            sub_image = sk_segmentation.find_boundaries(sub_image)

        # If multiple structures, add them together
        if len(structure_names) > 1:
            final_image = np.logical_or(final_image, sub_image)
        else:
            final_image = sub_image

    print("Converting image to 16 bit")
    final_image = scale_and_convert_to_16_bits(final_image)

    print("Saving image")
    brainio.to_nii(
        final_image,
        output_path,
        scale=atlas_scale,
        affine_transform=transformation_matrix,
    )

    print(f"Saved image at: {output_path}")


def main():
    start_time = datetime.now()
    print("Starting region volume generation")
    args = get_parser().parse_args()

    if args.atlas_config is None:
        args.atlas_config = source_custom_config()

    generate_region_volume(
        args.structure_names,
        args.atlas_path,
        args.output_filename,
        args.atlas_config,
        args.glass,
    )

    print("Finished. Total time taken: {}".format(datetime.now() - start_time))


if __name__ == "__main__":
    main()
