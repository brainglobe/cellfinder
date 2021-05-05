import logging

import numpy as np


def export_points(
    points,
    resolution,
    output_filename,
):
    logging.info("Exporting to brainrender")
    np.save(output_filename, points * resolution)
