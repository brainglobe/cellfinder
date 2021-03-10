"""
parser
========
All the various arguments that may be needed by the various submodules.
Defined together so they can also be called by any other entry points.
"""


from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    ArgumentTypeError,
)
from cellfinder import __version__
from imlib.general.numerical import check_positive_float, check_positive_int
from cellfinder_core.tools.source_files import source_custom_config_cellfinder

from cellfinder_core.download.cli import (
    model_parser,
    download_directory_parser,
)

from brainreg.cli import atlas_parse, geometry_parser, niftyreg_parse
from brainreg.cli import backend_parse as brainreg_backend_parse

# TODO: Gradually move all paths as strings to Path objects

models = {
    "18": "18-layer",
    "34": "34-layer",
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}


def valid_model_depth(depth):
    """
    Ensures a correct existing_model is chosen
    :param value: Input value
    :param models: Dict of allowed models
    :return: Input value, if it corresponds to a valid existing_model
    """

    if depth in models.keys():
        return depth
    else:
        raise ArgumentTypeError(
            f"Model depth: {depth} is not valid. Please "
            f"choose one of: {list(models.keys())}"
        )


def cellfinder_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = main_parse(parser)
    parser = config_parse(parser)
    parser = pixel_parser(parser)
    parser = run_parse(parser)
    parser = io_parse(parser)
    parser = cell_detect_parse(parser)
    parser = classification_parse(parser)
    parser = cube_extract_parse(parser)
    parser = misc_parse(parser)
    parser = model_parser(parser)
    parser = figures_parse(parser)
    parser = download_directory_parser(parser)

    # brainreg options
    parser = atlas_parse(parser)
    parser = geometry_parser(parser)
    parser = brainreg_backend_parse(parser)
    # This needs to be abstracted away into brainreg for multiple backends
    parser = niftyreg_parse(parser)

    return parser


def main_parse(parser):
    main_parser = parser.add_argument_group("General options")

    main_parser.add_argument(
        "-s",
        "--signal-planes-paths",
        dest="signal_planes_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path to the directory of the signal files. Can also be a text"
        "file pointing to the files.",
    )
    main_parser.add_argument(
        "-b",
        "--background-planes-path",
        dest="background_planes_path",
        type=str,
        nargs=1,
        required=True,
        help="Path to the directory of the background files. Can also be a "
        "text file pointing to the files.",
    )
    main_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for all intermediate and final results.",
    )
    main_parser.add_argument(
        "--signal-channel-ids",
        dest="signal_ch_ids",
        type=check_positive_int,
        nargs="+",
        help="Channel ID numbers, in the same order as 'signal-planes-paths'."
        " Will default to '0, 1, 2' etc, but maybe useful to specify.",
    )
    main_parser.add_argument(
        "--background-channel-id",
        dest="background_ch_id",
        type=check_positive_int,
        help="Channel ID number, corresponding to 'background-planes-path'.",
    )
    main_parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    return parser


def pixel_parser(parser):
    # TODO: separate the two groups. Image pixel sizes are needed in lots of
    # places that the network pixel sizes are not
    pixel_opt_parser = parser.add_argument_group(
        "Options to define pixel sizes of raw data"
    )
    pixel_opt_parser.add_argument(
        "-v",
        "--voxel-sizes",
        dest="voxel_sizes",
        required=True,
        nargs="+",
        help="Voxel sizes in microns, in the order of data orientation. "
        "e.g. '5 2 2'",
    )
    pixel_opt_parser.add_argument(
        "--network-voxel-sizes",
        dest="network_voxel_sizes",
        nargs="+",
        default=[5, 1, 1],
        help="Voxel sizes in microns that the machine learning network was "
        "trained on, in the order of data orientation. e.g. '5 2 2'."
        "Set this to adjust the pixel sizes of the extracted cubes",
    )
    return parser


def run_parse(parser):
    run_parser = parser.add_argument_group(
        "Options to disable part of cellfinder"
    )
    run_parser.add_argument(
        "--no-detection",
        dest="no_detection",
        action="store_true",
        help="Dont run cell candidate detection",
    )

    run_parser.add_argument(
        "--no-classification",
        dest="no_classification",
        action="store_true",
        help="Do not run cell classification",
    )
    run_parser.add_argument(
        "--no-register",
        dest="no_register",
        action="store_true",
        help="Do not perform registration",
    )
    run_parser.add_argument(
        "--no-analyse",
        dest="no_analyse",
        action="store_true",
        help="Do not analyse and export cell positions",
    )
    run_parser.add_argument(
        "--no-figures",
        dest="no_figures",
        action="store_true",
        help="Do not create figures (e.g. heatmap)",
    )

    return parser


def io_parse(parser):
    io_parser = parser.add_argument_group("Input & output options")
    io_parser.add_argument(
        "--start-plane",
        dest="start_plane",
        type=check_positive_int,
        default=0,
        help="The first plane to process in the Z dimension.",
    )
    io_parser.add_argument(
        "--end-plane",
        dest="end_plane",
        type=int,
        default=-1,
        help="The last plane to process in the Z dimension.",
    )
    return parser


def cell_detect_parse(parser):
    # TODO: improve the help on these files
    cell_detect_parser = parser.add_argument_group(
        "Detection options", description="Cell detection options"
    )
    cell_detect_parser.add_argument(
        "--save-planes",
        dest="save_planes",
        action="store_true",
        help="Whether to save the individual planes after "
        "processing and thresholding. Useful for debugging.",
    )

    cell_detect_parser.add_argument(
        "--outlier-keep",
        dest="outlier_keep",
        action="store_true",
        help="Dont remove putative cells that fall outside initial clusters",
    )

    cell_detect_parser.add_argument(
        "--artifact-keep",
        dest="artifact_keep",
        action="store_true",
        help="Save artifacts into the initial xml file",
    )

    cell_detect_parser.add_argument(
        "--max-cluster-size",
        dest="max_cluster_size",
        type=check_positive_int,
        default=100000,
        help="Largest putative cell cluster (in cubic um) where "
        "splitting should be attempted",
    )

    cell_detect_parser.add_argument(
        "--soma-diameter",
        dest="soma_diameter",
        type=check_positive_float,
        default=16,
        help="The expected soma size in um in the x/y dimensions",
    )

    cell_detect_parser.add_argument(
        "--ball-xy-size",
        dest="ball_xy_size",
        type=check_positive_int,
        default=6,
        help="The size in um of the ball used "
        "for the morphological filter in the x/y dimensions",
    )
    cell_detect_parser.add_argument(
        "--ball-z-size",
        dest="ball_z_size",
        type=check_positive_int,
        default=15,
        help="The size in um of the ball used "
        "for the morphological filter in the z dimension",
    )
    cell_detect_parser.add_argument(
        "--ball-overlap-fraction",
        dest="ball_overlap_fraction",
        type=check_positive_float,
        default=0.6,
        help="The fraction of the ball that has to cover thresholded pixels "
        "for the centre pixel to be considered a nucleus pixel",
    )

    cell_detect_parser.add_argument(
        "--log-sigma-size",
        dest="log_sigma_size",
        type=check_positive_float,
        default=0.2,
        help="The filter size used in the Laplacian of Gaussian filter to "
        "enhance the cell intensities. Given as a fraction of the "
        "soma-diameter.",
    )
    cell_detect_parser.add_argument(
        "--threshold",
        dest="n_sds_above_mean_thresh",
        type=check_positive_float,
        default=10,
        help="The cell threshold, in multiples of the standard deviation"
        "above the mean",
    )
    cell_detect_parser.add_argument(
        "--soma-spread-factor",
        dest="soma_spread_factor",
        type=check_positive_float,
        default=1.4,
        help="Soma size spread factor (for splitting up cell clusters)",
    )

    return parser


def classification_parse(parser):
    classification_parser = parser.add_argument_group(
        "Cell classification options"
    )
    classification_parser.add_argument(
        "--trained-model",
        dest="trained_model",
        type=str,
        help="Path to the trained model",
    )
    classification_parser.add_argument(
        "--model-weights",
        dest="model_weights",
        type=str,
        help="Path to existing model weights",
    )
    classification_parser.add_argument(
        "--network-depth",
        dest="network_depth",
        type=valid_model_depth,
        default="50",
        help="Resnet depth (based on He et al. (2015)",
    )
    classification_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=check_positive_int,
        default=32,
        help="Batch size for classification. Can be adjusted depending on "
        "GPU memory.",
    )
    return parser


def cube_extract_parse(parser):
    cube_extract_parser = parser.add_argument_group("Cube extraction options")
    cube_extract_parser.add_argument(
        "--cube-width",
        dest="cube_width",
        type=check_positive_int,
        default=50,
        help="The width of the cubes to extract (must be even)",
    )
    cube_extract_parser.add_argument(
        "--cube-height",
        dest="cube_height",
        type=check_positive_int,
        default=50,
        help="The height of the cubes to extract (must be even)",
    )
    cube_extract_parser.add_argument(
        "--cube-depth",
        dest="cube_depth",
        type=check_positive_int,
        default=20,
        help="The depth (z) of the cubes to extract",
    )
    cube_extract_parser.add_argument(
        "--save-empty-cubes",
        dest="save_empty_cubes",
        action="store_true",
        help="If a cube cannot be extracted (e.g. to close to the edge of"
        "the image), save an empty cube instead. Useful to keep track"
        "of all cell candidates.",
    )
    return parser


def config_parse(parser):
    config_opt_parser = parser.add_argument_group("Config options")
    config_opt_parser.add_argument(
        "--config",
        dest="registration_config",
        type=str,
        default=source_custom_config_cellfinder(),
        help="To supply your own, custom configuration file.",
    )

    return parser


def figures_parse(parser):
    figure_parser = parser.add_argument_group(
        "Figure generation specific parameters"
    )
    figure_parser.add_argument(
        "--heatmap-smoothing",
        dest="heatmap_smooth",
        type=check_positive_float,
        default=100,
        help="Gaussian smoothing sigma, in um.",
    )
    figure_parser.add_argument(
        "--no-mask-figs",
        dest="mask_figures",
        action="store_false",
        help="Don't mask the figures (removing any areas outside the brain,"
        "from e.g. smoothing)",
    )
    return parser


def misc_parse(parser):
    misc_parser = parser.add_argument_group("Misc options")
    misc_parser.add_argument(
        "--n-free-cpus",
        dest="n_free_cpus",
        type=check_positive_int,
        default=2,
        help="The number of CPU cores on the machine to leave "
        "unused by the program to spare resources.",
    )
    misc_parser.add_argument(
        "--max-ram",
        dest="max_ram",
        type=check_positive_float,
        default=None,
        help="Maximum amount of RAM to use (in GB) - not currently fully "
        "implemented for all parts of cellfinder",
    )
    misc_parser.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        help="Save .csv files of cell locations (in addition to xml)."
        "Useful for importing into other software.",
    )
    misc_parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Will increase verbosity of logging and save all "
        "intermediate files for diagnosis of software issues.",
    )
    misc_parser.add_argument(
        "--sort-input-file",
        dest="sort_input_file",
        action="store_true",
        help="If set to true, the input text file will be sorted using "
        "natural sorting. This means that the file paths will be "
        "sorted as would be expected by a human and "
        "not purely alphabetically",
    )
    return parser
