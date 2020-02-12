from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
from skimage.io import imread
from napari.utils.io import magic_imread
from imlib.general.system import get_sorted_file_paths

from imlib.IO.cells import cells_xml_to_df
from imlib.cells.cells import Cell


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="img_paths", type=str, help="Directory of images")
    parser.add_argument(
        dest="cells_xml", type=str, help="Path to the .xml cell file"
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


def cells_df_as_np(cells_df, new_order=[2, 1, 0], type_column="type"):
    cells_df = cells_df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()
    return cells


def get_cell_arrays(cells_file):
    df = cells_xml_to_df(cells_file)

    non_cells = df[df["type"] == Cell.UNKNOWN]
    cells = df[df["type"] == Cell.CELL]

    cells = cells_df_as_np(cells)
    non_cells = cells_df_as_np(non_cells)
    return cells, non_cells


def estimate_image_max(image_paths, multiplier=2):
    centre_plane = int(len(image_paths) / 2)
    max_value = imread(image_paths[centre_plane]).max()
    return int(multiplier * max_value)


def main():
    args = parser().parse_args()
    img_paths = get_sorted_file_paths(args.img_paths, file_extension=".tif")
    cells, non_cells = get_cell_arrays(args.cells_xml)

    with napari.gui_qt():
        v = napari.Viewer(title="Cellfinder cell viewer")
        images = magic_imread(img_paths, use_dask=True, stack=True)
        max_value = estimate_image_max(img_paths)
        v.add_image(images, contrast_limits=[0, max_value])
        v.add_points(
            non_cells,
            size=args.marker_size,
            n_dimensional=True,
            opacity=args.opacity,
            symbol=args.symbol,
            face_color="lightskyblue",
            name="Non-Cells",
        )
        v.add_points(
            cells,
            size=args.marker_size,
            n_dimensional=True,
            opacity=args.opacity,
            symbol=args.symbol,
            face_color="lightgoldenrodyellow",
            name="Cells",
        )


if __name__ == "__main__":
    main()
