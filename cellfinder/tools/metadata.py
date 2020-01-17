"""
metadata
===============

Functions to read image acqusition metadata files.

"""

from cellfinder.tools.exceptions import CommandLineInputError
from argparse import ArgumentTypeError

from micrometa.micrometa import get_acquisition_metadata


def define_pixel_sizes(args):
    """
    Defines the pixel sizes based on metadata file, or CLI flags
    :param args: Cellfinder argument object
    :return: args updated
    """

    if None in [args.x_pixel_um, args.y_pixel_um, args.z_pixel_um]:
        if args.metadata is None:
            raise CommandLineInputError(
                "Not all pixel sizes are defined on the command line, but "
                "no metadata file has been supplied"
            )
        try:
            metadata = get_acquisition_metadata(args.metadata)
        except NotImplementedError:
            raise CommandLineInputError(
                "Not all pixel sizes are defined on the command line, but the"
                " metadata file cannot be read."
            )

    for dim in ["x", "y", "z"]:
        dim_attribute = "{}_pixel_um".format(dim)
        if getattr(args, dim_attribute) is None:
            try:
                setattr(args, dim_attribute, getattr(metadata, dim_attribute))
            except ArgumentTypeError:
                raise CommandLineInputError(
                    "No {} pixel size was defined on the command line, and it "
                    "cannot be parsed from the metadata".format(dim)
                )

    return args
