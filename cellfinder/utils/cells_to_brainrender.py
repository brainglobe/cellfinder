import argparse
from cellfinder.IO import cells as cells_io

from cellfinder.tools.misc import check_positive_float, check_positive_int

# Temporary utility until cellfinder -> brainrender data flow is finalised


def cells_to_brainrender(
    cells_file,
    output_filename,
    pixel_size_x=10,
    pixel_size_y=10,
    pixel_size_z=10,
    max_z=13200,
    key="df",
):
    print(f"Converting file: {cells_file}")
    cells = cells_io.get_cells(cells_file)
    cells = cells_io.cells_to_dataframe(cells)

    cells["x"] = cells["x"] * pixel_size_x
    cells["y"] = cells["y"] * pixel_size_y
    cells["z"] = cells["z"] * pixel_size_z

    cells.columns = ["z", "y", "x", "type"]

    cells["x"] = max_z - cells["x"]

    print(f"Saving to: {output_filename}")
    cells.to_hdf(output_filename, key=key, mode="w")

    print("Finished")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="cells_file",
        type=str,
        help="Cellfinder cells file to be converted",
    )
    parser.add_argument(
        dest="output_filename",
        type=str,
        help="Output filename. Should end with '.h5'",
    )

    parser.add_argument(
        "-x",
        "--x-pixel-size",
        dest="x_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "-y",
        "--y-pixel-size",
        dest="y_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "-z",
        "--z-pixel-size",
        dest="z_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "--max-z",
        dest="max_z",
        type=check_positive_int,
        default=13200,
        help="Maximum z extent of the atlas",
    )
    parser.add_argument(
        "--hdf-key",
        dest="hdf_key",
        type=str,
        default="df",
        help="hdf identifier ",
    )
    return parser


def main():
    args = get_parser().parse_args()
    cells_to_brainrender(
        args.cells_file,
        args.output_filename,
        pixel_size_x=args.x_pixel_size,
        pixel_size_y=args.y_pixel_size,
        pixel_size_z=args.z_pixel_size,
        max_z=args.max_z,
        key=args.hdf_key,
    )


if __name__ == "__main__":
    main()
