import argparse
import os
import logging

import pandas as pd
from tqdm import tqdm


from brainio import brainio

from imlib.IO.cells import get_cells
from imlib.pandas.misc import sanitise_df
from imlib.image.metadata import define_pixel_sizes
from imlib.general.config import get_config_obj
from imlib.IO.structures import load_structures_as_df

from imlib.anatomy.structures.structures_tree import (
    atlas_value_to_structure_id,
    CellCountMissingCellsException,
    UnknownAtlasValue,
)

import cellfinder.tools.parser as cellfinder_parse
from cellfinder.tools.prep import prep_atlas_conf, Paths
from cellfinder.tools.source_files import get_structures_path

LEFT_HEMISPHERE = 2
RIGHT_HEMISPHERE = 1


def region_summary_cli_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = cli_parse(parser)
    parser = cellfinder_parse.count_summary_parse(parser)
    parser = cellfinder_parse.atlas_parse(parser)
    parser = cellfinder_parse.pixel_parser(parser)
    return parser


def cli_parse(parser):
    cli_parser = parser.add_argument_group("Input data options")
    cli_parser.add_argument(
        "--registered-atlas",
        dest="registered_atlas_path",
        type=str,
        help="The path to the atlas registered to the sample brain",
    )
    cli_parser.add_argument(
        "--hemispheres",
        dest="hemispheres_atlas_path",
        type=str,
        help="The atlas with just the hemispheres encoded.",
    )
    cli_parser.add_argument(
        "--xml",
        dest="xml_file_path",
        type=str,
        help="The xml file containing the cell locations",
    )

    cli_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for all intermediate and final results.",
    )

    return parser


def get_cells_data(xml_file_path, cells_only=True):
    cells = get_cells(xml_file_path, cells_only=cells_only)
    if not cells:
        raise CellCountMissingCellsException(
            "No cells found in file: {}".format(xml_file_path)
        )
    return cells


def get_scales(sample_pixel_sizes, atlas_pixel_sizes, scale=True):
    if scale:
        sample_x, sample_y, sample_z = sample_pixel_sizes
        atlas_x_pix_size = atlas_pixel_sizes["x"]
        atlas_y_pix_size = atlas_pixel_sizes["y"]
        atlas_z_pix_size = atlas_pixel_sizes["z"]

        x_scale = float(sample_x) / float(atlas_x_pix_size)
        y_scale = float(sample_y) / float(atlas_y_pix_size)
        z_scale = float(sample_z) / float(atlas_z_pix_size)
        return x_scale, y_scale, z_scale
    else:
        return 1, 1, 1


def get_atlas_pixel_sizes(atlas_config_path):
    config_obj = get_config_obj(atlas_config_path)
    atlas_conf = config_obj["atlas"]
    atlas_pixel_sizes = atlas_conf["pixel_size"]
    return atlas_pixel_sizes


def get_max_coords(cells):
    max_x, max_y, max_z = (0, 0, 0)
    for cell in cells:
        if cell.x > max_x:
            max_x = cell.x
        if cell.y > max_y:
            max_y = cell.y
        if cell.z > max_z:
            max_z = cell.z
    return max_x, max_y, max_z


def transform_cell_coords(atlas, cell, scales):
    x_scale, y_scale, z_scale = scales
    # convert to atlas coordinates
    cell.soft_transform(x_scale, y_scale, z_scale, integer=True)
    # In case we reorientated the sample, not the atlas.
    flip_dims = False  # FIXME: put CLI option
    if flip_dims:
        cell.transformed_y, cell.transformed_z = (
            cell.transformed_z,
            cell.transformed_y,
        )  # TODO: do in cell
        # TEST: check that correct dim to flip
        cell.transformed_z = atlas.shape[2] - cell.transformed_z


def get_cells_nbs_df(cells, structures_reference_df, structures_with_cells):
    structures_with_cells = list(structures_with_cells)
    cell_numbers = pd.DataFrame(
        columns=("structure_name", "hemisphere", "cell_count")
    )
    for structure in structures_with_cells:
        for hemisphere in (1, 2):
            n_cells = len(
                [
                    c
                    for c in cells
                    if c.structure_id == structure
                    and c.hemisphere == hemisphere
                ]
            )
            if n_cells:
                struct_name = structures_reference_df[
                    structures_reference_df["structure_id_path"] == structure
                ]["name"].values[0]
                cell_numbers = cell_numbers.append(
                    {
                        "structure_name": struct_name,
                        "hemisphere": hemisphere,
                        "cell_count": n_cells,
                    },
                    ignore_index=True,
                )
    sorted_cell_numbers = cell_numbers.sort_values(
        by=["cell_count"], ascending=False
    )
    return sorted_cell_numbers


def get_structure_from_coordinates(
    atlas, cell, max_coords, order=(0, 1, 2), structures_reference_df=None
):
    transformed_coords = (
        cell.transformed_x,
        cell.transformed_y,
        cell.transformed_z,
    )
    try:
        atlas_value = atlas[
            transformed_coords[order[0]],
            transformed_coords[order[1]],
            transformed_coords[order[2]],
        ]
    except IndexError as err:
        logging.warning(
            "The cell {}, scaled to {} "
            "falls outside of the atlas with "
            "dimensions {}. Treating as outside the brain".format(
                cell, transformed_coords, atlas.shape, max_coords, err
            )
        )
        return 0

    if structures_reference_df is None:
        return atlas_value
    else:
        try:
            structure_id = atlas_value_to_structure_id(
                atlas_value, structures_reference_df
            )
        except UnknownAtlasValue as err:
            print(
                "Skipping cell {} (scaled: {}), missing value {}".format(
                    cell, transformed_coords, err
                )
            )
            return
        else:
            return structure_id


def analysis_run(args, file_name="summary_cell_counts.csv"):
    args = prep_atlas_conf(args)

    if args.structures_file_path is None:
        args.structures_file_path = get_structures_path()

    atlas = brainio.load_any(args.paths.registered_atlas_path)
    hemisphere = brainio.load_any(args.paths.hemispheres_atlas_path)

    cells = get_cells_data(
        args.paths.classification_out_file, cells_only=args.cells_only,
    )
    max_coords = get_max_coords(cells)  # Useful for debugging dimensions
    structures_reference_df = load_structures_as_df(args.structures_file_path)

    atlas_pixel_sizes = get_atlas_pixel_sizes(args.atlas_config)
    sample_pixel_sizes = args.x_pixel_um, args.y_pixel_um, args.z_pixel_um

    scales = get_scales(
        sample_pixel_sizes, atlas_pixel_sizes, args.scale_cell_coordinates
    )

    structures_with_cells = set()
    for i, cell in enumerate(tqdm(cells)):
        transform_cell_coords(atlas, cell, scales)

        structure_id = get_structure_from_coordinates(
            atlas,
            cell,
            max_coords,
            order=args.coordinates_order,
            structures_reference_df=structures_reference_df,
        )
        if structure_id is not None:
            cell.structure_id = structure_id

            structures_with_cells.add(structure_id)
        else:
            continue

        cell.hemisphere = get_structure_from_coordinates(
            hemisphere, cell, max_coords, order=args.coordinates_order
        )

    sorted_cell_numbers = get_cells_nbs_df(
        cells, structures_reference_df, structures_with_cells
    )

    combined_hemispheres = combine_df_hemispheres(sorted_cell_numbers)
    df = calculate_densities(combined_hemispheres, args.paths.volume_csv_path)
    df = sanitise_df(df)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, file_name)
    df.to_csv(output_file, index=False)


def calculate_densities(counts, volume_csv_path):
    """
    Use the region volume information from registration to calculate cell
    densities. Based on the atlas names, which must be exactly equal.
    :param counts: dataframe with cell counts
    :param volume_csv_path: path of the volumes of each brain region
    :return:
    """
    volumes = pd.read_csv(volume_csv_path, sep=",", header=0, quotechar='"')
    df = pd.merge(counts, volumes, on="structure_name", how="outer")
    df = df.fillna(0)
    df["left_cells_per_mm3"] = df.left_cell_count / df.left_volume_mm3
    df["right_cells_per_mm3"] = df.right_cell_count / df.right_volume_mm3
    return df


def combine_df_hemispheres(df):
    """
    Combine left and right hemisphere data onto a single row
    :param df:
    :return:
    """
    left = df[df["hemisphere"] == LEFT_HEMISPHERE]
    right = df[df["hemisphere"] == RIGHT_HEMISPHERE]
    left = left.drop(["hemisphere"], axis=1)
    right = right.drop(["hemisphere"], axis=1)
    left.rename(columns={"cell_count": "left_cell_count"}, inplace=True)
    right.rename(columns={"cell_count": "right_cell_count"}, inplace=True)
    both = pd.merge(left, right, on="structure_name", how="outer")
    both = both.fillna(0)
    both["total_cells"] = both.left_cell_count + both.right_cell_count
    both = both.sort_values("total_cells", ascending=False)
    return both


def main():
    args = region_summary_cli_parser().parse_args()
    args = define_pixel_sizes(args)
    args.paths = Paths(args, args.output_dir)
    args.paths.registered_atlas_path = args.registered_atlas_path
    args.paths.hemispheres_atlas_path = args.hemispheres_atlas_path
    args.paths.classification_out_file = args.xml_file_path
    analysis_run(args)


if __name__ == "__main__":
    main()
