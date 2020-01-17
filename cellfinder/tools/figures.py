import logging
import numpy as np

import cellfinder.IO.cells as cell_io
from cellfinder.cells.cells import Cell


def get_bins(image_size, bin_sizes):
    """
    Given an image size, and bin size, return a list of the bin boundaries
    :param image_size: Size of the final image (tuple/list)
    :param bin_sizes: Bin sizes corresponding to the dimensions of
    "image_size" (tuple/list)
    :return: List of arrays of bin boundaries
    """
    bins = []
    for dim in range(0, len(image_size)):
        bins.append(np.arange(0, image_size[dim], bin_sizes[dim]))
    return bins


def get_cell_location_array(
    cell_file,
    cell_position_scaling=[None, None, None],
    cells_only=False,
    type_str="type",
    integer=True,
):
    """
    Loads a cell file, and converts to an array, with 3 columns of x,y,z
    positions
    :param cell_file: Any supported cell file, e.g. xml
    :param cell_position_scaling: list of cell scaling (raw -> final) for
    [x, y, z]
    :param cells_only: If only cells (rather than unknown or artifacts)
    should be included
    :param str type_str: String defining the title of the cell type column
    in the dataframe. Used to remove non cells (artifacts), and then to clean
    up the dataframe to be converted into a numpy array.
    :param integer: Force integer cell positions (default: True)
    :return: Array of cell positions, with x,y,z columns
    """

    logging.debug("Loading cells")
    cells = cell_io.get_cells(cell_file)

    if cell_position_scaling != [None, None, None]:
        for cell in cells:
            cell.transform(
                x_scale=cell_position_scaling[0],
                y_scale=cell_position_scaling[1],
                z_scale=cell_position_scaling[2],
                integer=integer,
            )

    cells = cell_io.cells_to_dataframe(cells)
    num_cells = len(cells[cells[type_str] == Cell.CELL])
    num_non_cells = len(cells[cells[type_str] == Cell.NO_CELL])
    logging.debug(
        "{} cells, and {} non-cells".format(num_cells, num_non_cells)
    )
    if cells_only:
        logging.debug("Removing non cells")
        cells = cells[cells[type_str] == Cell.CELL]

    logging.debug("Tidying up dataframe to convert to array")
    cells.drop(type_str, axis=1, inplace=True)
    return cells.to_numpy()
