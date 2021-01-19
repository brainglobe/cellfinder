from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
import numpy as np
from napari.utils.io import magic_imread
from pathlib import Path

from qtpy.QtWidgets import QApplication

from imlib.general.system import get_sorted_file_paths, ensure_directory_exists
from imlib.general.list import unique_elements_lists

from imlib.IO.cells import cells_xml_to_df, save_cells, get_cells
from imlib.cells.cells import Cell
from imlib.IO.yaml import save_yaml

from cellfinder.extract.extract_cubes import main as extract_cubes_main
import cellfinder.tools.parser as cellfinder_parse

OUTPUT_NAME = "curated_cells.xml"
CURATED_POINTS = []


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = curation_parser(parser)
    parser = cellfinder_parse.pixel_parser(parser)
    parser = cellfinder_parse.misc_parse(parser)
    parser = cellfinder_parse.cube_extract_parse(parser)
    return parser


def curation_parser(parser):
    parser.add_argument(
        dest="signal_image_paths", type=str, help="Signal images"
    )
    parser.add_argument(
        dest="background_image_paths",
        type=str,
        help="Background images",
    )

    parser.add_argument(
        dest="cells_xml", type=str, help="Path to the .xml cell file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for curation results",
    )
    parser.add_argument(
        "--symbol", type=str, default="ring", help="Marker symbol."
    )
    parser.add_argument(
        "--marker-size", type=int, default=15, help="Marker size."
    )
    parser.add_argument(
        "--opacity", type=float, default=0.6, help="Opacity of the markers."
    )
    return parser


def get_cell_labels_arrays(
    cells_file, new_order=[2, 1, 0], type_column="type"
):
    df = cells_xml_to_df(cells_file)

    labels = df[type_column]
    labels = labels.to_numpy()
    cells_df = df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()

    # convert to boolean
    labels = labels == 2
    return cells, labels


def main():
    args = parser().parse_args()
    # args = define_pixel_sizes(args)

    if args.output is None:
        output = Path(args.cells_xml)
        output_directory = output.parent
        print(
            f"No output directory given, so setting output "
            f"directory to: {output_directory}"
        )
    else:
        output_directory = Path(args.output)

    ensure_directory_exists(str(output_directory))
    output_filename = output_directory / OUTPUT_NAME

    img_paths = get_sorted_file_paths(
        args.signal_image_paths, file_extension=".tif"
    )
    cells, labels = get_cell_labels_arrays(args.cells_xml)

    properties = {"cell": labels}

    with napari.gui_qt():
        viewer = napari.Viewer(title="Cellfinder cell curation")
        images = magic_imread(img_paths, use_dask=True, stack=True)
        viewer.add_image(images)
        face_color_cycle = ["lightskyblue", "lightgoldenrodyellow"]
        viewer.add_points(
            cells,
            properties=properties,
            symbol=args.symbol,
            n_dimensional=True,
            size=args.marker_size,
            face_color="cell",
            face_color_cycle=face_color_cycle,
            name="Cell candidates",
        )

        @viewer.bind_key("t")
        def toggle_point_property(viewer):
            """Toggle point type"""
            selected_points = list(viewer.layers[1].selected_data)
            if selected_points:
                selected_properties = viewer.layers[1].properties["cell"][
                    selected_points
                ]
                toggled_properties = np.logical_not(selected_properties)
                viewer.layers[1].properties["cell"][
                    selected_points
                ] = toggled_properties

                # Add curated cells to list
                CURATED_POINTS.extend(selected_points)
                print(
                    f"{len(selected_points)} points "
                    f"toggled and added to the list "
                )

                # refresh the properties colour
                viewer.layers[1].refresh_colors(update_color_mapping=False)

        @viewer.bind_key("c")
        def confirm_point_property(viewer):
            """Confirm point type"""
            selected_points = list(viewer.layers[1].selected_data)
            if selected_points:
                # Add curated cells to list
                CURATED_POINTS.extend(selected_points)
                print(
                    f"{len(selected_points)} points "
                    f"confirmed and added to the list "
                )

        @viewer.bind_key("Alt-Q")
        def save_curation(viewer):
            """Save file"""
            if not CURATED_POINTS:
                print("No cells have been confirmed or toggled, not saving")
            else:
                unique_cells = unique_elements_lists(CURATED_POINTS)
                points = viewer.layers[1].data[unique_cells]
                labels = viewer.layers[1].properties["cell"][unique_cells]
                labels = labels.astype("int")
                labels = labels + 1

                cells_to_save = []
                for idx, point in enumerate(points):
                    cell = Cell([point[2], point[1], point[0]], labels[idx])
                    cells_to_save.append(cell)

                print(f"Saving results to: {output_filename}")
                save_cells(cells_to_save, output_filename)
                print("Done!")

        @viewer.bind_key("Alt-E")
        def start_cube_extraction(viewer):
            """Extract cubes for training"""

            if not output_filename.exists():
                print(
                    "No curation results have been saved. "
                    "Please save before extracting cubes"
                )
            else:
                print(f"Saving cubes to: {output_directory}")
                run_extraction(
                    output_filename,
                    output_directory,
                    args.signal_image_paths,
                    args.background_image_paths,
                    args.cube_depth,
                    args.cube_width,
                    args.cube_height,
                    args.voxel_sizes,
                    args.network_voxel_sizes,
                    args.max_ram,
                    args.n_free_cpus,
                    args.save_empty_cubes,
                )

                print("Saving yaml file to use for training")
                save_yaml_file(output_directory)

                print("Closing window")
                QApplication.closeAllWindows()
                print(
                    "Finished! You may now annotate more "
                    "datasets, or go straight to training"
                )


def run_extraction(
    output_filename,
    output_directory,
    signal_paths,
    background_paths,
    cube_depth,
    cube_width,
    cube_height,
    voxel_sizes,
    network_voxel_sizes,
    max_ram,
    n_free_cpus,
    save_empty_cubes,
):
    planes_paths = {}
    planes_paths[0] = get_sorted_file_paths(
        signal_paths, file_extension=".tif"
    )
    planes_paths[1] = get_sorted_file_paths(
        background_paths, file_extension=".tif"
    )

    all_candidates = get_cells(str(output_filename))

    cells = [c for c in all_candidates if c.is_cell()]
    non_cells = [c for c in all_candidates if not c.is_cell()]

    to_extract = {"cells": cells, "non_cells": non_cells}

    for cell_type, cell_list in to_extract.items():
        print(f"Extracting type: {cell_type}")
        cell_type_output_directory = output_directory / cell_type
        print(f"Saving to: {cell_type_output_directory}")
        ensure_directory_exists(str(cell_type_output_directory))
        extract_cubes_main(
            cell_list,
            cell_type_output_directory,
            planes_paths,
            cube_depth,
            cube_width,
            cube_height,
            voxel_sizes,
            network_voxel_sizes,
            max_ram,
            n_free_cpus,
            save_empty_cubes,
        )


def save_yaml_file(output_directory):
    yaml_filename = output_directory / "training.yml"
    yaml_section = [
        {
            "cube_dir": str(output_directory / "cells"),
            "cell_def": "",
            "type": "cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
        {
            "cube_dir": str(output_directory / "non_cells"),
            "cell_def": "",
            "type": "no_cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
    ]

    yaml_contents = {"data": yaml_section}
    save_yaml(yaml_contents, yaml_filename)


if __name__ == "__main__":
    main()
