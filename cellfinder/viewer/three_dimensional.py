from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from brainio import brainio
import napari


def parser_3d_view():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        dest="image",
        type=str,
        help="Can be the path of a nifti file, tiff file, tiff files folder "
        "or text file containing a list of paths",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Load planes in parallel using "
        "multiprocessing for faster data loading.",
    )
    return parser


def main():
    args = parser_3d_view().parse_args()
    data = brainio.load_any(args.image, load_parallel=args.parallel)
    with napari.gui_qt():
        v = napari.Viewer(title="Cellfinder 3D viewer", ndisplay=3)
        v.add_image(data)


if __name__ == "__main__":
    main()
