from typing import Dict

import numpy as np
from numba import jit


def get_biggest_structure(sizes):
    result = 0
    for val in sizes:
        if val > result:
            result = val
    return result


class BaseTileFilter:
    def __init__(self, out_of_brain_intensity_threshold=100):
        """

        :param int out_of_brain_intensity_threshold: Set to 0 to disable
        """
        self.out_of_brain_intensity_threshold = (
            out_of_brain_intensity_threshold
        )
        self.current_threshold = -1
        self.keep = True
        self.size_analyser = SizeAnalyser()

    def set_tile(self, tile):
        raise NotImplementedError

    def get_tile(self):
        raise NotImplementedError

    def get_structures(self):
        struct_sizes = []
        self.size_analyser.process(self._tile, self.current_threshold)
        struct_sizes = self.size_analyser.get_sizes()
        return get_biggest_structure(struct_sizes), struct_sizes.size()


@jit
def is_low_average(tile: np.ndarray, threshold: float) -> bool:
    """
    Return `True` if the average value of *tile* is below *threshold*.
    """
    avg = np.mean(tile)
    return avg < threshold


class OutOfBrainTileFilter(BaseTileFilter):
    def set_tile(self, tile):
        self._tile = tile

    def get_tile(self):
        return self._tile


class SizeAnalyser:
    obsolete_ids: Dict[int, int] = {}
    struct_sizes: Dict[int, int] = {}

    def process(self, tile, threshold):
        tile = tile.copy()
        self.clear_maps()

        last_structure_id = 1

        for y in range(tile.shape[1]):
            for x in range(tile.shape[0]):
                # default struct_id to 0 so that it is not counted as
                # structure in next iterations
                id_west = id_north = struct_id = 0
                if tile[x, y] >= threshold:
                    # If in bounds look to neighbours
                    if x > 0:
                        id_west = tile[x - 1, y]
                    if y > 0:
                        id_north = tile[x, y - 1]

                    id_west = self.sanitise_id(id_west)
                    id_north = self.sanitise_id(id_north)

                    if id_west != 0:
                        if id_north != 0 and id_north != id_west:
                            struct_id = self.merge_structures(
                                id_west, id_north
                            )
                        else:
                            struct_id = id_west
                    elif id_north != 0:
                        struct_id = id_north
                    else:  # no neighbours, create new structure
                        struct_id = last_structure_id
                        self.struct_sizes[last_structure_id] = 0
                        last_structure_id += 1

                    self.struct_sizes[struct_id] += 1

                tile[x, y] = struct_id

    def get_sizes(self):
        for iterator_pair in self.struct_sizes:
            self.struct_sizes.push_back(iterator_pair.second)
        return self.struct_sizes

    def clear_maps(self):
        self.obsolete_ids.clear()
        self.struct_sizes.clear()

    def sanitise_id(self, s_id):
        while self.obsolete_ids.count(
            s_id
        ):  # walk up the chain of obsolescence
            s_id = self.obsolete_ids[s_id]
        return s_id

    # id1 and id2 must be valid struct IDs (>0)!
    def merge_structures(self, id1, id2):
        # ensure id1 is the smaller of the two values
        if id2 < id1:
            tmp_id = id1  # swap
            id1 = id2
            id2 = tmp_id

        self.struct_sizes[id1] += self.struct_sizes[id2]
        self.struct_sizes.erase(id2)

        self.obsolete_ids[id2] = id1

        return id1
