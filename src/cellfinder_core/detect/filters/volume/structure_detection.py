from dataclasses import dataclass

import numpy as np
from numba import jit
from numba.core import types
from numba.typed import Dict


@dataclass
class Point:
    x: int
    y: int
    z: int


UINT64_MAX = np.iinfo(np.uint64).max
N_NEIGHBOURS_4_CONNECTED = 3  # top left, below
N_NEIGHBOURS_8_CONNECTED = 13  # all the 9 below + the 4 before on same plane


@jit
def get_non_zero_ull_min(values):
    min_val = UINT64_MAX
    for v in values:
        if v != 0 and v < min_val:
            min_val = v
    return min_val


def get_non_zero_ull_min_wrapper(values):  # wrapper for testing purposes
    assert len(values) == 10
    return get_non_zero_ull_min(values)


def get_structure_centre(structure):
    mean_x = 0
    mean_y = 0
    mean_z = 0
    s_len = len(structure)

    for p in structure:
        mean_x += p.x / s_len
        mean_y += p.y / s_len
        mean_z += p.z / s_len

    return Point(round(mean_x), round(mean_y), round(mean_z))


def get_structure_centre_wrapper(structure):  # wrapper for testing purposes
    s = []
    for p in structure:
        if type(p) == dict:
            s.append(Point(p["x"], p["y"], p["z"]))
        else:
            s.append(Point(p.x, p.y, p.z))
    return get_structure_centre(s)


class CellDetector:
    def __init__(self, width: int, height: int, start_z: int, connect_type=4):
        self.shape = width, height
        self.z = start_z

        if connect_type not in (4, 8):
            raise ValueError("Connection type must be one of [4, 8]")
        self.connect_type = connect_type

        self.SOMA_CENTRE_VALUE = UINT64_MAX

        # position to append in stack
        # FIXME: replace by keeping start_z and self.z > self.start_Z
        self.relative_z = 0
        self.next_structure_id = 1

        self.structure_manager = StructureManager()

    def get_previous_layer(self):
        return np.array(self.previous_layer, dtype=np.uint64)

    def process(
        self, layer
    ):  # WARNING: inplace  # WARNING: ull may be overkill but ulong required
        if [e for e in layer.shape[:2]] != [e for e in self.shape]:
            raise ValueError("layer does not have correct shape")

        source_dtype = layer.dtype
        layer = layer.astype(np.uint64)

        # The 'magic numbers' below are chosen so that the maximum number
        # representable in each data type is converted to 2**64 - 1, the
        # maximum representable number in uint64.
        if source_dtype == np.uint8:
            # 2**56 + 2**48 + 2**40 + 2**32 + 2**24 + 2**16 + 2**8 + 1
            layer *= 72340172838076673  # TEST:
        elif source_dtype == np.uint16:
            # 2**48 + 2**32 + 2**16 + 1
            layer *= 281479271743489
        elif source_dtype == np.uint32:
            # 2**32 + 1
            layer *= 4294967297
        elif source_dtype == np.uint64:
            pass
        else:
            raise ValueError("Layer does not have uint data type")

        if self.connect_type == 4:
            self.previous_layer = self.connect_four(layer)
        else:
            self.previous_layer = self.connect_eight(layer)

        if self.relative_z == 0:
            self.relative_z += 1

        self.z += 1

    def connect_four(self, layer):
        """
        For all the pixels in the current layer, finds all structures touching
        this pixel using the
        four connected (plus shape) rule and also looks at the pixel at the
        same location in the previous layer.
        If structures are found, they are added to the structure manager and
        the pixel labeled accordingly.

        :param layer:
        :return:
        """
        # Labels of structures at left, top, below
        neighbour_ids = [0] * N_NEIGHBOURS_4_CONNECTED

        for y in range(layer.shape[1]):
            for x in range(layer.shape[0]):
                if layer[x, y] == self.SOMA_CENTRE_VALUE:
                    for i in range(N_NEIGHBOURS_4_CONNECTED):  # reset
                        neighbour_ids[
                            i
                        ] = 0  # Labels of structures at left, top, below
                    # If in bounds look at neighbours
                    if x > 0:
                        neighbour_ids[0] = layer[x - 1, y]
                    if y > 0:
                        neighbour_ids[1] = layer[x, y - 1]
                    if self.relative_z > 0:
                        neighbour_ids[2] = self.previous_layer[x, y]

                    if is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.structure_manager.add(
                        x, y, self.z, neighbour_ids
                    )
                else:
                    # reset so that grayscale value does not count as
                    # structure in next iterations
                    struct_id = 0

                layer[x, y] = struct_id
        return layer

    def connect_eight(self, layer):
        """
        For all the pixels in the current layer, finds all structures touching
        this pixel using the
        eight connected (connected by edges or corners) rule and also looks at
        the pixel at the same
        location in the previous layer.
        If structures are found, they are added to the structure manager and
        the pixel labeled accordingly.

        :param layer:
        :return:
        """
        # Labels of neighbour structures touching before
        neighbour_ids = [0] * N_NEIGHBOURS_8_CONNECTED

        for y in range(layer.shape[1]):
            for x in range(layer.shape[0]):
                if layer[x, y] == self.SOMA_CENTRE_VALUE:
                    for i in range(N_NEIGHBOURS_8_CONNECTED):  # reset
                        neighbour_ids[i] = 0

                    # If in bounds look at neighbours
                    if x > 0 and y > 0:
                        neighbour_ids[0] = layer[x - 1, y - 1]
                    if x > 0:
                        neighbour_ids[1] = layer[x - 1, y]
                    if y > 0:
                        neighbour_ids[2] = layer[x, y - 1]
                        neighbour_ids[3] = layer[x + 1, y - 1]
                    if self.relative_z > 0:
                        if x > 0 and y > 0:
                            neighbour_ids[4] = self.previous_layer[
                                x - 1, y - 1
                            ]
                        if x > 0:
                            neighbour_ids[5] = self.previous_layer[x - 1, y]
                            if y < layer.shape[1] - 1:
                                neighbour_ids[6] = self.previous_layer[
                                    x - 1, y + 1
                                ]
                        if y > 0:
                            neighbour_ids[7] = self.previous_layer[x, y - 1]
                            if x < layer.shape[0] - 1:
                                neighbour_ids[8] = self.previous_layer[
                                    x + 1, y - 1
                                ]
                        neighbour_ids[9] = self.previous_layer[x, y]
                        if y < layer.shape[1] - 1:
                            neighbour_ids[10] = self.previous_layer[x, y + 1]
                        if x < layer.shape[0] - 1:
                            neighbour_ids[11] = self.previous_layer[x + 1, y]
                            if y < layer.shape[1] - 1:
                                neighbour_ids[12] = self.previous_layer[
                                    x + 1, y + 1
                                ]

                    if is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.structure_manager.add(
                        x, y, self.z, neighbour_ids
                    )
                else:
                    # reset so that grayscale value does not count as
                    # structure in next iterations
                    struct_id = 0

                layer[x, y] = struct_id
        return layer

    def get_cell_centres(self):
        cell_centres = self.structure_manager.structures_to_cells()
        return cell_centres

    def get_coords_list(self):
        # TODO: cache (attribute)
        coords_arrays = self.structure_manager.get_coords_dict()
        # Convert from arrays to dicts
        coords = {}
        for sid in coords_arrays:
            coords[sid] = []
            for row in coords_arrays[sid]:
                coords[sid].append({"x": row[0], "y": row[1], "z": row[2]})
        return coords


class StructureManager:
    def __init__(self):
        # Mapping from obsolete IDs to the IDs that they have been
        # made obsolete by
        self.obsolete_ids = Dict.empty(
            key_type=types.int64, value_type=types.uint64
        )
        # Mapping from IDs to list of points in that structure
        self.coords_maps = Dict.empty(
            key_type=types.int64, value_type=types.uint64[:, :]
        )

    def get_coords_dict(self):
        return self.coords_maps

    def add_point(self, sid: int, point: np.ndarray) -> None:
        """
        Add *point* to the structure with the given *sid*.
        """
        point = np.array(point).astype(np.uint64)
        self.coords_maps[sid] = np.row_stack((self.coords_maps[sid], point))

    def add(self, x: int, y: int, z: int, neighbour_ids: list[int]) -> int:
        """
        For the current coordinates takes all the neighbours and find the
        minimum structure including obsolete structures mapping to any of
        the neighbours recursively.

        Once the correct structure id is found, append a point with the
        current coordinates to the coordinates map entry for the correct
        structure. Hence each entry of the map will be a vector of all the
        pertaining points.
        """
        updated_id = self.sanitise_ids(neighbour_ids)
        if updated_id not in self.coords_maps:
            self.coords_maps[updated_id] = np.zeros(
                shape=(0, 3), dtype=np.uint64
            )
        self.merge_structures(updated_id, neighbour_ids)

        # Add point for that structure
        self.add_point(updated_id, [x, y, z])
        return updated_id

    def sanitise_ids(self, neighbour_ids: list[int]) -> int:
        """
        Get the smallest ID of all the structures that are connected to IDs
        in `neighbour_ids`.

        For all the neighbour ids, walk up the chain of obsolescence (self.
        obsolete_ids) to reassign the corresponding most obsolete structure
        to the current neighbour.

        Has no side effects on this class.
        """
        for i, neighbour_id in enumerate(neighbour_ids):
            # walk up the chain of obsolescence
            while neighbour_id in self.obsolete_ids:
                neighbour_id = self.obsolete_ids[neighbour_id]
            neighbour_ids[i] = neighbour_id

        # Get minimum of all non-obsolete IDs
        updated_id = get_non_zero_ull_min(neighbour_ids)
        return updated_id

    def merge_structures(
        self, updated_id: int, neighbour_ids: list[int]
    ) -> None:
        """
        For all the neighbours, reassign all the points of neighbour to
        updated_id. Then deletes the now obsolete entry from the points
        map and add that entry to the obsolete_ids.

        Updates:
        - self.coords_maps
        - self.obsolete_ids
        """
        for i, neighbour_id in enumerate(neighbour_ids):
            # minimise ID so if neighbour with higher ID, reassign its points
            # to current
            if neighbour_id > updated_id:
                for p in self.coords_maps[neighbour_id]:
                    self.add_point(updated_id, p)
                self.coords_maps.pop(neighbour_id)
                self.obsolete_ids[neighbour_id] = updated_id

    def structures_to_cells(self):
        cell_centres = []
        for iterator_pair in self.coords_maps:
            structure = iterator_pair.second
            p = get_structure_centre(structure)
            cell_centres.append(p)
        return cell_centres


@jit
def is_new_structure(neighbour_ids):
    for i in range(len(neighbour_ids)):
        if neighbour_ids[i] != 0:
            return False
    return True
