from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
from napari.utils.io import magic_imread
from pathlib import Path
from cellfinder.IO.cells import cells_xml_to_df, save_cells
from cellfinder.cells.cells import Cell
from cellfinder.tools.system import get_sorted_file_paths
import numpy as np
from cellfinder.tools.tools import unique_elements_lists

OUTPUT_NAME = "curated_cells.xml"

# based on:
# https://github.com/kevinyamauchi/napari/blob/points_data_refactor/
# examples/add_points_with_annotations.py
## only in napari feature branch currently


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="img_paths", type=str, help="Directory of images")
    parser.add_argument(
        dest="cells_xml", type=str, help="Path to the .xml cell file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for curation results",
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

    if args.output is None:
        output = Path(args.cells_xml)
        OUTPUT_FILEPATH = output.parent / OUTPUT_NAME
        print(
            f"No output directory given, so setting output "
            f"directory to: {OUTPUT_FILEPATH}"
        )
    else:
        OUTPUT_FILEPATH = args.output

    img_paths = get_sorted_file_paths(args.img_paths, file_extension=".tif")
    cells, labels = get_cell_labels_arrays(args.cells_xml)

    annotations = {"cell": labels}

    CURATED_CELLS = []

    with napari.gui_qt():
        viewer = napari.Viewer(title="Cellfinder cell viewer")
        images = magic_imread(img_paths, use_dask=True, stack=True)

        viewer.add_image(images)
        face_color_cycle = ["lightskyblue", "lightgoldenrodyellow"]
        points_layer = viewer.add_points(
            cells,
            annotations=annotations,
            symbol=args.symbol,
            n_dimensional=True,
            size=args.marker_size,
            face_color="cell",
            face_color_cycle=face_color_cycle,
        )

        # bind a function to toggle the good_point annotation of the
        # selected points
        @viewer.bind_key("t")
        def toggle_point_annotation(viewer):
            selected_points = viewer.layers[1].selected_data
            if selected_points:
                selected_annotations = viewer.layers[1].annotations["cell"][
                    selected_points
                ]
                toggled_annotations = np.logical_not(selected_annotations)
                viewer.layers[1].annotations["cell"][
                    selected_points
                ] = toggled_annotations

                # Add curated cells to list
                CURATED_CELLS.extend(selected_points)
                print(
                    f"{len(selected_points)} points "
                    f"toggled and added to the list "
                )

                # we need to manually refresh since we did not use
                # the Points.annotations setter
                points_layer._refresh_face_color()

        @viewer.bind_key("c")
        def confirm_point_annotation(viewer):
            selected_points = viewer.layers[1].selected_data
            if selected_points:
                # Add curated cells to list
                CURATED_CELLS.extend(selected_points)
                print(
                    f"{len(selected_points)} points "
                    f"confirmed and added to the list "
                )

        @viewer.bind_key("Control-S")
        def save_curation(viewer):
            if CURATED_CELLS == []:
                print("No cells have been confirmed or toggled, not saving")
            else:
                unique_cells = unique_elements_lists(CURATED_CELLS)
                points = viewer.layers[1].data[unique_cells]
                labels = viewer.layers[1].annotations["cell"][unique_cells]
                labels = labels.astype("int")
                labels = labels + 1

                cells_to_save = []
                for idx, point in enumerate(points):
                    cell = Cell([point[2], point[1], point[0]], labels[idx])
                    cells_to_save.append(cell)
                print(f"Saving results to: {OUTPUT_FILEPATH}")
                save_cells(cells_to_save, OUTPUT_FILEPATH)


if __name__ == "__main__":
    main()
