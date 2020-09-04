import logging

import pandas as pd


def export_points(
    points,
    atlas,
    output_directory,
    resolution,
    name="points",
    points_file_extension=".h5",
):
    logging.info("Exporting to brainrender")
    max_axis_2 = atlas.metadata["shape"][2]
    output_filename = output_directory / (name + points_file_extension)
    points = pd.DataFrame(points * resolution)
    points.columns = ["x", "y", "z"]
    # BR is oriented differently (for now)
    points["z"] = (max_axis_2 * resolution) - points["z"]
    points.to_hdf(output_filename, key="df", mode="w")
