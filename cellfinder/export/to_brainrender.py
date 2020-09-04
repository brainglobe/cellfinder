import logging

import pandas as pd


def export_points(
    points,
    atlas,
    resolution,
    output_filename,
):
    logging.info("Exporting to brainrender")
    max_axis_2 = atlas.metadata["shape"][2]
    points = pd.DataFrame(points * resolution)
    points.columns = ["x", "y", "z"]
    # BR is oriented differently (for now)
    points["z"] = (max_axis_2 * resolution) - points["z"]
    points.to_hdf(output_filename, key="df", mode="w")
