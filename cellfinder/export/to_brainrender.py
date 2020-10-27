import logging

import pandas as pd


def export_points(
    points, resolution, output_filename,
):
    logging.info("Exporting to brainrender")
    points = pd.DataFrame(points * resolution)
    points.columns = ["x", "y", "z"]
    points.to_hdf(output_filename, key="df", mode="w")
