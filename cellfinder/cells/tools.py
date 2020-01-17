from collections import defaultdict


def group_cells_by_z(cells):
    """
    For a list of Cells return a dict of lists of cells, grouped by plane.

    :param list cells: list of cells from cellfinder.cells.cells.Cell
    :return:  default
    dict, with each key being a plane (e.g. 1280) and each entry being a list
    of Cells
    """
    cells_groups = defaultdict(list)
    for cell in cells:
        cells_groups[cell.z].append(cell)
    return cells_groups


class MissingCellsError(Exception):
    pass
