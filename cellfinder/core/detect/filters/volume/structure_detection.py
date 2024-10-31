from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TypeVar, Union

import numba.typed
import numpy as np
import numpy.typing as npt
from numba import njit, objmode, typed
from numba.core import types
from numba.experimental import jitclass
from numba.types import DictType

from cellfinder.core.tools.tools import get_max_possible_int_value

T = TypeVar("T")
# type used for the domain of the volume - the size of the vol
vol_np_type = np.int64
vol_numba_type = types.int64
# type used for the structure id
sid_np_type = np.uint64
sid_numba_type = types.uint64


@dataclass
class Point:
    x: int
    y: int
    z: int


@njit
def get_non_zero_dtype_min(values: np.ndarray) -> sid_numba_type:
    """
    Get the minimum of non-zero entries in *values*.

    If all entries are zero, returns maximum storeable number
    in the values array.
    """
    # we don't know how big the int is, so make it as large as possible (64)
    with objmode(min_val="u8"):
        min_val = get_max_possible_int_value(values.dtype)

    for v in values:
        if v != 0 and v < min_val:
            min_val = v
    return min_val


@njit
def traverse_dict(d: Dict[T, T], a: T) -> T:
    """
    Traverse d, until a is not present as a key.
    """
    value = a
    while value in d:
        value = d[value]
    return value


@njit
def get_structure_centre(structure: np.ndarray) -> np.ndarray:
    """
    Get the pixel coordinates of the centre of a structure.

    Centre calculated as the mean of each pixel coordinate,
    rounded to the nearest integer.
    """
    # numba support axis for sum, but not mean
    return np.round(np.sum(structure, axis=0) / structure.shape[0])


@njit
def _get_structure_centre(structure: types.ListType) -> np.ndarray:
    # See get_structure_centre.
    # this is for our own points stored as list optimized by numba
    a_sum = 0.0
    b_sum = 0.0
    c_sum = 0.0
    for a, b, c in structure:
        a_sum += a
        b_sum += b
        c_sum += c

    return np.round(
        np.array(
            [
                a_sum / len(structure),
                b_sum / len(structure),
                c_sum / len(structure),
            ]
        )
    )


# Type declaration has to come outside of the class,
# see https://github.com/numba/numba/issues/8808
tuple_point_type = types.Tuple(
    (vol_numba_type, vol_numba_type, vol_numba_type)
)
list_of_points_type = types.ListType(tuple_point_type)


spec = [
    ("z", vol_numba_type),
    ("next_structure_id", sid_numba_type),
    ("soma_centre_value", sid_numba_type),  # as large as possible
    ("shape", types.UniTuple(vol_numba_type, 2)),
    ("obsolete_ids", DictType(sid_numba_type, sid_numba_type)),
    ("coords_maps", DictType(sid_numba_type, list_of_points_type)),
]


@jitclass(spec=spec)
class CellDetector:
    """
    A class to detect connected structures within a series of
    stacked planes.

    Attributes
    ----------
    z :
        z-index of the plane currently being processed.
    next_structure_id :
        The next available structure ID that has yet to be used. IDs start
        counting up from 1.
    shape :
        Shape of the planes to be processed.
    obsolete_ids :
        Mapping from obsolete structure IDs to the structure ID they
        have been replaced with. This is used to keep track of structures
        that were initially disconnected, but later connected as the planes
        are scanned.
    coords_maps :
        Mapping from structure ID to the coordinates of pixels within that
        structure. Coordinates are stored in a list of (x, y, z) tuples of
        the coordinates.

        Use `get_structures` to get it as a dict whose values are each
        a 2D array, where rows are points, and columns x, y, z of the
        points.
    """

    def __init__(
        self,
        height: int,
        width: int,
        start_z: int,
        soma_centre_value: sid_numba_type,
    ):
        """
        Parameters
        ----------
        height, width:
            Shape of the planes input to self.process()
        start_z:
            The z-coordinate of the first processed plane.
        """
        self.shape = height, width
        self.z = start_z
        self.next_structure_id = 1
        self.soma_centre_value = soma_centre_value

        # Mapping from obsolete IDs to the IDs that they have been
        # made obsolete by
        self.obsolete_ids = numba.typed.Dict.empty(
            key_type=sid_numba_type, value_type=sid_numba_type
        )
        # Mapping from IDs to list of points in that structure
        self.coords_maps = numba.typed.Dict.empty(
            key_type=sid_numba_type, value_type=list_of_points_type
        )

    def _set_soma(self, soma_centre_value: sid_numba_type):
        # Due to https://github.com/numba/numba/issues/9576. For testing we try
        # different data types. Because of that issue we cannot pass a uint64
        # soma_centre_value to constructor after we pass a uint32. This is the
        # only way for now until numba fixes the issue
        self.soma_centre_value = soma_centre_value

    def process(
        self, plane: np.ndarray, previous_plane: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Process a new plane (should be in Y, X axis order).
        """
        if plane.shape[:2] != self.shape:
            raise ValueError("plane does not have correct shape")

        plane = self.connect_four(plane, previous_plane)
        self.z += 1
        return plane

    def connect_four(
        self, plane: np.ndarray, previous_plane: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Perform structure labelling.

        For all the pixels in the current plane, finds all structures touching
        this pixel using the four connected (plus shape) rule and also looks at
        the pixel at the same location in the previous plane. If structures are
        found, they are added to the structure manager and the pixel labeled
        accordingly.

        Returns
        -------
        plane :
            Plane with pixels either set to zero (no structure) or labelled
            with their structure ID. Plane is in Y, X axis order.
        """
        soma_centre_value = self.soma_centre_value
        for y in range(plane.shape[0]):
            for x in range(plane.shape[1]):
                if plane[y, x] == soma_centre_value:
                    # Labels of structures below, left and behind
                    neighbour_ids = np.zeros(3, dtype=sid_np_type)
                    # If in bounds look at neighbours
                    if y > 0:
                        neighbour_ids[0] = plane[y - 1, x]
                    if x > 0:
                        neighbour_ids[1] = plane[y, x - 1]
                    if previous_plane is not None:
                        neighbour_ids[2] = previous_plane[y, x]

                    if is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.add(x, y, self.z, neighbour_ids)
                else:
                    # reset so that grayscale value does not count as
                    # structure in next iterations
                    struct_id = 0

                plane[y, x] = struct_id

        return plane

    def get_cell_centres(self) -> np.ndarray:
        """
        Returns the 2D array of cell centers. It's (N, 3) with X, Y, Z columns.
        """
        return self.structures_to_cells()

    def get_structures(self) -> Dict[int, np.ndarray]:
        """
        Gets the structures as a dict of structure IDs mapped to the 2D array
        of structure points (points vs x, y, z columns).
        """
        d = {}
        for sid, points in self.coords_maps.items():
            # numba silliness - it cannot handle
            # `item = np.array(points, dtype=vol_np_type)` so we need to create
            # array and then fill in the point
            item = np.empty((len(points), 3), dtype=vol_np_type)
            # need to cast to int64, otherwise when dict is used we can get
            # warnings as in numba issue #8829 b/c it assumes it's uint64.
            # Python uses int(64) as the type
            d[types.int64(sid)] = item

            for i, point in enumerate(points):
                item[i, :] = point

        return d

    def add_point(
        self, sid: int, point: Union[tuple, list, np.ndarray]
    ) -> None:
        """
        Add single 3d (x, y, z) *point* to the structure with the given *sid*.
        """
        # cast in case user passes in int64 (default type for int in python)
        # and numba complains
        key = sid_numba_type(sid)
        if key not in self.coords_maps:
            self.coords_maps[key] = typed.List.empty_list(tuple_point_type)

        self._add_point(key, (int(point[0]), int(point[1]), int(point[2])))

    def add_points(self, sid: int, points: np.ndarray):
        """
        Adds ndarray of *points* to the structure with the given *sid*.
        Each row is a 3-column (x, y, z) point.
        """
        # cast in case user passes in int64 (default type for int in python)
        # and numba complains
        key = sid_numba_type(sid)
        if key not in self.coords_maps:
            self.coords_maps[key] = typed.List.empty_list(tuple_point_type)

        append = self.coords_maps[key].append
        pts = np.round(points).astype(vol_np_type)
        for point in pts:
            append((point[0], point[1], point[2]))

    def _add_point(
        self, sid: sid_numba_type, point: Tuple[int, int, int]
    ) -> None:
        # sid must exist
        self.coords_maps[sid].append(point)

    def add(
        self, x: int, y: int, z: int, neighbour_ids: npt.NDArray[sid_np_type]
    ) -> sid_numba_type:
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
            self.coords_maps[updated_id] = typed.List.empty_list(
                tuple_point_type
            )
        self.merge_structures(updated_id, neighbour_ids)

        # Add point for that structure
        self._add_point(updated_id, (int(x), int(y), int(z)))
        return updated_id

    def sanitise_ids(
        self, neighbour_ids: npt.NDArray[sid_np_type]
    ) -> sid_numba_type:
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
            neighbour_id = traverse_dict(self.obsolete_ids, neighbour_id)
            neighbour_ids[i] = neighbour_id

        # Get minimum of all non-obsolete IDs
        updated_id = get_non_zero_dtype_min(neighbour_ids)
        return updated_id

    def merge_structures(
        self,
        updated_id: sid_numba_type,
        neighbour_ids: npt.NDArray[sid_np_type],
    ) -> None:
        """
        For all the neighbours, reassign all the points of neighbour to
        updated_id. Then deletes the now obsolete entry from the points
        map and add that entry to the obsolete_ids.

        Updates:
        - self.coords_maps
        - self.obsolete_ids
        """
        for neighbour_id in np.unique(neighbour_ids):
            # minimise ID so if neighbour with higher ID, reassign its points
            # to current
            if neighbour_id > updated_id:
                self.coords_maps[updated_id].extend(
                    self.coords_maps[neighbour_id]
                )
                self.coords_maps.pop(neighbour_id)
                self.obsolete_ids[neighbour_id] = updated_id

    def structures_to_cells(self) -> np.ndarray:
        cell_centres = np.empty((len(self.coords_maps), 3))
        for idx, structure in enumerate(self.coords_maps.values()):
            p = _get_structure_centre(structure)
            cell_centres[idx] = p
        return cell_centres


@njit
def is_new_structure(neighbour_ids: np.ndarray) -> bool:
    for i in range(len(neighbour_ids)):
        if neighbour_ids[i] != 0:
            return False
    return True
