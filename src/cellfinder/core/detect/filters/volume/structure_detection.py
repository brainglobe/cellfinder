from dataclasses import dataclass
from typing import Dict, Optional, TypeVar

import numba.typed
import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core import types
from numba.experimental import jitclass
from numba.types import DictType


@dataclass
class Point:
    x: int
    y: int
    z: int


@njit
def get_non_zero_dtype_min(values: np.ndarray) -> int:
    """
    Get the minimum of non-zero entries in *values*.

    If all entries are zero, returns maximum storeable number
    in the values array.
    """
    min_val = np.iinfo(values.dtype).max
    for v in values:
        if v != 0 and v < min_val:
            min_val = v
    return min_val


T = TypeVar("T")


@njit
def traverse_dict(d: Dict[T, T], a: T) -> T:
    """
    Traverse d, until a is not present as a key.
    """
    if a in d:
        return traverse_dict(d, d[a])
    else:
        return a


@njit
def get_structure_centre(structure: np.ndarray) -> np.ndarray:
    """
    Get the pixel coordinates of the centre of a structure.

    Centre calculated as the mean of each pixel coordinate,
    rounded to the nearest integer.
    """
    # can't do np.mean(structure, axis=0)
    # because axis is not supported by numba
    return np.round(
        np.array(
            [
                np.mean(structure[:, 0]),
                np.mean(structure[:, 1]),
                np.mean(structure[:, 2]),
            ]
        )
    )


# Type declaration has to come outside of the class,
# see https://github.com/numba/numba/issues/8808
uint_2d_type = types.uint64[:, :]


spec = [
    ("z", types.uint64),
    ("next_structure_id", types.uint64),
    ("shape", types.UniTuple(types.int64, 2)),
    ("obsolete_ids", DictType(types.int64, types.int64)),
    ("coords_maps", DictType(types.uint64, uint_2d_type)),
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
        structure. Coordinates are stored in a 2D array, with the second
        axis indexing (x, y, z) coordinates.
    """

    def __init__(self, width: int, height: int, start_z: int):
        """
        Parameters
        ----------
        width, height
            Shape of the planes input to self.process()
        start_z:
            The z-coordinate of the first processed plane.
        """
        self.shape = width, height
        self.z = start_z
        self.next_structure_id = 1

        # Mapping from obsolete IDs to the IDs that they have been
        # made obsolete by
        self.obsolete_ids = numba.typed.Dict.empty(
            key_type=types.int64, value_type=types.int64
        )
        # Mapping from IDs to list of points in that structure
        self.coords_maps = numba.typed.Dict.empty(
            key_type=types.int64, value_type=uint_2d_type
        )

    def process(
        self, plane: np.ndarray, previous_plane: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Process a new plane.
        """
        if [e for e in plane.shape[:2]] != [e for e in self.shape]:
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
            with their structure ID.
        """
        SOMA_CENTRE_VALUE = np.iinfo(plane.dtype).max
        for y in range(plane.shape[1]):
            for x in range(plane.shape[0]):
                if plane[x, y] == SOMA_CENTRE_VALUE:
                    # Labels of structures below, left and behind
                    neighbour_ids = np.zeros(3, dtype=np.uint64)
                    # If in bounds look at neighbours
                    if x > 0:
                        neighbour_ids[0] = plane[x - 1, y]
                    if y > 0:
                        neighbour_ids[1] = plane[x, y - 1]
                    if previous_plane is not None:
                        neighbour_ids[2] = previous_plane[x, y]

                    if is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.add(x, y, self.z, neighbour_ids)
                else:
                    # reset so that grayscale value does not count as
                    # structure in next iterations
                    struct_id = 0

                plane[x, y] = struct_id

        return plane

    def get_cell_centres(self) -> np.ndarray:
        return self.structures_to_cells()

    def get_coords_dict(self) -> Dict:
        return self.coords_maps

    def add_point(self, sid: int, point: np.ndarray) -> None:
        """
        Add *point* to the structure with the given *sid*.
        """
        self.coords_maps[sid] = np.row_stack((self.coords_maps[sid], point))

    def add(
        self, x: int, y: int, z: int, neighbour_ids: npt.NDArray[np.uint64]
    ) -> int:
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
        point = np.array([[x, y, z]], dtype=np.uint64)
        self.add_point(updated_id, point)
        return updated_id

    def sanitise_ids(self, neighbour_ids: npt.NDArray[np.uint64]) -> int:
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
            neighbour_id = int(traverse_dict(self.obsolete_ids, neighbour_id))
            neighbour_ids[i] = neighbour_id

        # Get minimum of all non-obsolete IDs
        updated_id = get_non_zero_dtype_min(neighbour_ids)
        return int(updated_id)

    def merge_structures(
        self, updated_id: int, neighbour_ids: npt.NDArray[np.uint64]
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
                self.add_point(updated_id, self.coords_maps[neighbour_id])
                self.coords_maps.pop(neighbour_id)
                self.obsolete_ids[neighbour_id] = updated_id

    def structures_to_cells(self) -> np.ndarray:
        cell_centres = np.empty((len(self.coords_maps.keys()), 3))
        for idx, structure in enumerate(self.coords_maps.values()):
            p = get_structure_centre(structure)
            cell_centres[idx] = p
        return cell_centres


@njit
def is_new_structure(neighbour_ids: np.ndarray) -> bool:
    for i in range(len(neighbour_ids)):
        if neighbour_ids[i] != 0:
            return False
    return True
