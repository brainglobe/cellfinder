import os
import argparse

import pandas as pd
from brainio import brainio
from imlib.general.numerical import check_positive_float, check_positive_int

import cellfinder.summarise.count_summary as cells_regions
from imlib.IO.cells import cells_to_xml
from cellfinder.tools.prep import prep_atlas_conf
from cellfinder.tools.source_files import get_structures_path
from cellfinder.summarise.structures.structures_tree import (
    load_structures_as_df,
)


def xml_crop(args, df_query="name"):
    args = prep_atlas_conf(args)

    if args.reference_structures_file_path is None:
        args.reference_structures_file_path = get_structures_path()
    if args.structures_file_path is None:
        args.structures_file_path = get_structures_path()

    reference_struct_df = pd.read_csv(args.reference_structures_file_path)
    curate_struct_df = pd.read_csv(args.structures_file_path)

    curate_struct_df = reference_struct_df[
        reference_struct_df[df_query].isin(curate_struct_df[df_query])
    ]

    curated_ids = list(curate_struct_df["structure_id_path"])

    atlas = brainio.load_any(args.registered_atlas_path)
    hemisphere = brainio.load_any(args.hemispheres_atlas_path)

    structures_reference_df = load_structures_as_df(
        args.reference_structures_file_path
    )

    atlas_pixel_sizes = cells_regions.get_atlas_pixel_sizes(args.atlas_config)
    sample_pixel_sizes = args.x_pixel_um, args.y_pixel_um, args.z_pixel_um

    scales = cells_regions.get_scales(sample_pixel_sizes, atlas_pixel_sizes)

    destination_folder = os.path.join(args.xml_dir, "xml_crop")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    xml_names = [f for f in os.listdir(args.xml_dir) if f.endswith(".xml")]
    xml_paths = [os.path.join(args.xml_dir, f) for f in xml_names]

    for idx, xml_path in enumerate(xml_paths):
        print("Curating file: {}".format(xml_names[idx]))
        cells = cells_regions.get_cells_data(
            xml_path, cells_only=args.cells_only,
        )
        max_coords = cells_regions.get_max_coords(cells)

        curated_cells = []
        for i, cell in enumerate(cells):
            cells_regions.transform_cell_coords(atlas, cell, scales)

            structure_id = cells_regions.get_structure_from_coordinates(
                atlas,
                cell,
                max_coords,
                order=args.coordinates_order,
                structures_reference_df=structures_reference_df,
            )
            if structure_id in curated_ids:
                if args.hemisphere_query in [1, 2]:
                    hemisphere = cells_regions.get_structure_from_coordinates(
                        hemisphere,
                        cell,
                        max_coords,
                        order=args.coordinates_order,
                    )
                    if hemisphere is args.hemisphere_query:
                        curated_cells.append(cell)
                else:
                    curated_cells.append(cell)
        cells_to_xml(
            curated_cells,
            os.path.join(destination_folder, xml_names[idx]),
            artifact_keep=True,
        )
    print("Done!")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--xml-dir",
        dest="xml_dir",
        type=str,
        required=True,
        help="Directory containing XML files to be cropped",
    )

    parser.add_argument(
        "--structures-file",
        dest="structures_file_path",
        type=str,
        help="Curated csv structure list (as per the allen brain atlas csv",
    )
    parser.add_argument(
        "--ref-structures-file",
        dest="reference_structures_file_path",
        type=str,
        help="The csv file containing the structures "
        "definition (if not using the default "
        "Allen brain atlas).",
    )
    parser.add_argument(
        "--hemisphere-query",
        dest="hemisphere_query",
        type=int,
        default=0,
        help="Which hemisphere to keep (1 or 2). Default: 0 (all)",
    )

    parser.add_argument(
        "--registered-atlas",
        dest="registered_atlas_path",
        type=str,
        help="The path to the atlas registered to the sample brain",
    )
    parser.add_argument(
        "--hemispheres",
        dest="hemispheres_atlas_path",
        type=str,
        help="The atlas with just the hemispheres encoded.",
    )
    parser.add_argument(
        "--atlas-config",
        dest="atlas_config",
        type=str,
        help="Atlas configuration file. In the same format as the"
        "registration config file",
    )
    parser.add_argument(
        "--cells-only",
        dest="cells_only",
        action="store_false",
        help="Used for testing. Will include non cells in the checks",
    )

    parser.add_argument(
        "--coordinates-order",
        dest="coordinates_order",
        nargs=3,
        type=check_positive_int,
        default=[0, 1, 2],
        help="The order in which to read the dimensions in the atlas from the"
        " cell coordinates. 0,1,2 means x,y,z. 1,0,2 means y,x,z",
    )
    parser.add_argument(
        "-x",
        "--x-pixel-um",
        dest="x_pixel_um",
        type=check_positive_float,
        default=1,
        help="Pixel spacing of the data in the first "
        "dimension, specified in mm.",
    )
    parser.add_argument(
        "-y",
        "--y-pixel-um",
        dest="y_pixel_um",
        type=check_positive_float,
        default=1,
        help="Pixel spacing of the data in the second "
        "dimension, specified in mm.",
    )
    parser.add_argument(
        "-z",
        "--z-pixel-mm",
        dest="z_pixel_um",
        type=check_positive_float,
        default=5,
        help="Pixel spacing of the data in the third "
        "dimension, specified in mm.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    xml_crop(args)


if __name__ == "__main__":
    main()
