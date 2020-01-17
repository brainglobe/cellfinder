import numpy as np
import cellfinder.summarise.count_summary as cells_regions


def get_transformation_matrix(atlas_config):
    """
    From an atlas config, return transformation_matrix for proper nifti output
    :param atlas_config:
    :return: transformation_matrix
    """
    atlas_pixel_sizes = cells_regions.get_atlas_pixel_sizes(atlas_config)
    transformation_matrix = np.eye(4)
    for i, axis in enumerate(("x", "y", "z")):
        transformation_matrix[i, i] = atlas_pixel_sizes[axis]
    return transformation_matrix
