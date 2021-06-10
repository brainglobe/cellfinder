import logging
import numpy as np


def export_points(
    point_info,
    resolution,
    output_filename,
):
    logging.info("Exporting to abc4d")

    point_arrays = []
    for point in point_info:
        point.atlas_coordinate = point.atlas_coordinate * resolution
        point_arrays.append(
            np.append(point.atlas_coordinate, point.structure_id)
        )

    np.save(output_filename, np.vstack(point_arrays).astype(np.float64))
