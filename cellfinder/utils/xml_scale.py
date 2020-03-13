import argparse

from pathlib import Path
from datetime import datetime
from imlib.general.numerical import check_positive_float
from imlib.general.system import ensure_directory_exists
import imlib.IO.cells as cell_io

from imlib.general.exceptions import CommandLineInputError


def xml_scale(
    xml_file,
    x_scale=1,
    y_scale=1,
    z_scale=1,
    output_directory=None,
    integer=True,
):
    # TODO: add a csv option

    """
    To rescale the cell positions within an XML file. For compatibility with
    other software, or if  data has been scaled after cell detection.
    :param xml_file: Any cellfinder xml file
    :param x_scale: Rescaling factor in the first dimension
    :param y_scale: Rescaling factor in the second dimension
    :param z_scale: Rescaling factor in the third dimension
    :param output_directory: Directory to save the rescaled XML file.
    Defaults to the same directory as the input XML file
    :param integer: Force integer cell positions (default: True)
    :return:
    """
    if x_scale == y_scale == z_scale == 1:
        raise CommandLineInputError(
            "All rescaling factors are 1, " "please check the input."
        )
    else:
        input_file = Path(xml_file)
        start_time = datetime.now()
        cells = cell_io.get_cells(xml_file)

        for cell in cells:
            cell.transform(
                x_scale=x_scale,
                y_scale=y_scale,
                z_scale=z_scale,
                integer=integer,
            )

        if output_directory:
            output_directory = Path(output_directory)
        else:
            output_directory = input_file.parent

        ensure_directory_exists(output_directory)
        output_filename = output_directory / (input_file.stem + "_rescaled")
        output_filename = output_filename.with_suffix(input_file.suffix)

        cell_io.save_cells(cells, output_filename)

        print(
            "Finished. Total time taken: {}".format(
                datetime.now() - start_time
            )
        )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="xml_file", type=str, help="XML file to be scaled"
    )

    parser.add_argument(
        "-x",
        "--x-scale",
        dest="x_scale",
        type=check_positive_float,
        default=1,
        help="Rescaling factor in the first dimension.",
    )
    parser.add_argument(
        "-y",
        "--y-scale",
        dest="y_scale",
        type=check_positive_float,
        default=1,
        help="Rescaling factor in the second dimension.",
    )
    parser.add_argument(
        "-z",
        "--z-scale",
        dest="z_scale",
        type=check_positive_float,
        default=1,
        help="Rescaling factor in the third dimension.",
    )
    parser.add_argument(
        "-o" "--output",
        dest="output_directory",
        type=str,
        default=None,
        help="Directory to save the rescaled XML file. Defaults to the same "
        "directory as the input XML file.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    xml_scale(
        args.xml_file,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        z_scale=args.z_scale,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
