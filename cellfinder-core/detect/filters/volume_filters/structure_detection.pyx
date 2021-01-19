# distutils: language = c++

# cython: language_level=3

import cython

cimport libc.math as cmath

from libcpp.vector cimport vector as CppVector
from libcpp.map cimport map as CppMap
from libcpp.pair cimport pair as CppPair

import numpy as np

from cellfinder.detect.filters.typedefs cimport ull, uint, Point

from imlib.cells.cells import Cell



DEF ULLONG_MAX = 18446744073709551615 # (2**64) -1
DEF N_NEIGHBOURS_4_CONNECTED = 3  # top left, below
DEF N_NEIGHBOURS_8_CONNECTED = 13  # all the 9 below + the 4 before on same plane



cdef get_non_zero_ull_min(ull[:] values):
    cdef ull min_val = ULLONG_MAX
    cdef ull s_id
    cdef uint i
    for i in range(len(values)):
        s_id = values[i]
        if s_id != 0:
            if s_id < min_val:
                min_val = s_id
    return min_val

cpdef get_non_zero_ull_min_wrapper(values):  # wrapper for testing purposes
    assert len(values) == 10
    cdef ull c_values[10]
    for i, v in enumerate(values):
        c_values[i] = <ull> v
    return get_non_zero_ull_min(c_values)

cdef get_structure_centre(CppVector[Point] structure):
    cdef double mean_z, mean_y, mean_x
    mean_x = 0; mean_y = 0; mean_z = 0
    cdef double s_len = len(structure)

    cdef Point p
    for p in structure:
        mean_x += p.x / s_len
        mean_y += p.y / s_len
        mean_z += p.z / s_len

    return Point(<ull> cmath.round(mean_x), <ull> cmath.round(mean_y), <ull> cmath.round(mean_z))


cpdef get_structure_centre_wrapper(structure):  # wrapper for testing purposes
    cdef CppVector[Point] s
    for p in structure:
        if type(p) == dict:
            s.push_back(Point(p['x'], p['y'], p['z']))
        else:
            s.push_back(Point(p.x, p.y, p.z))
    return get_structure_centre(s)



cdef class CellDetector:
    cdef:
        public ull SOMA_CENTRE_VALUE  # = range - 1 (e.g. 2**16 - 1)

        ull z
        int relative_z
        int connect_type

        ull[:,:] previous_layer
        tuple shape

        StructureManager structure_manager

        ull next_structure_id

    def __init__(self, uint width, uint height, uint start_z, connect_type=4):
        self.shape = width, height
        self.z = start_z

        assert connect_type in (4, 8), 'Connection type must be one of 4,8 got "{}"'.format(connect_type)
        self.connect_type = connect_type

        self.SOMA_CENTRE_VALUE = ULLONG_MAX

        self.relative_z = 0  # position to append in stack  # FIXME: replace by keeping start_z and self.z > self.start_Z
        self.next_structure_id = 1

        self.structure_manager = StructureManager()

    cpdef get_previous_layer(self):
        return np.array(self.previous_layer, dtype=np.uint64)

    cpdef process(self, layer):  # WARNING: inplace  # WARNING: ull may be overkill but ulong required
        assert [e for e in layer.shape[:2]] == [e for e in self.shape], \
            'CellDetector layer error, expected shape "{}", got "{}"'\
                .format(self.shape, [e for e in layer.shape[:2]])

        source_dtype = layer.dtype
        layer = layer.astype(np.uint64)

        cdef ull[:,:] c_layer

        if source_dtype == np.uint8:
            layer *= 72340172838076656  # TEST:
        elif source_dtype == np.uint16:
            layer *= 281479271743489
        elif source_dtype == np.uint32:
            layer *= 4294967297
        elif source_dtype == np.uint64:
            pass
        else:
            raise ValueError('Expected layer of any type from np.uint8, np.uint16, np.uint32, np.uint64,'
                             'got: {}'.format(source_dtype))
        c_layer = layer

        if self.connect_type == 4:
            self.previous_layer = self.connect_four(c_layer)
        else:
            self.previous_layer = self.connect_eight(c_layer)

        if self.relative_z == 0:
            self.relative_z += 1

        self.z += 1

    @cython.boundscheck(False)
    cdef connect_four(self, ull[:,:] layer):
        """
        For all the pixels in the current layer, finds all structures touching this pixel using the
        four connected (plus shape) rule and also looks at the pixel at the same location in the previous layer.
        If structures are found, they are added to the structure manager and the pixel labeled accordingly.

        :param layer:
        :return:
        """
        cdef ull struct_id
        cdef ull neighbour_ids[N_NEIGHBOURS_4_CONNECTED]
        cdef uint i
        for i in range(N_NEIGHBOURS_4_CONNECTED):  # reset
            neighbour_ids[i] = 0 # Labels of structures at left, top, below

        cdef uint y, x
        for y in range(layer.shape[1]):
            for x in range(layer.shape[0]):
                if layer[x, y] == self.SOMA_CENTRE_VALUE:
                    for i in range(N_NEIGHBOURS_4_CONNECTED):  # reset
                        neighbour_ids[i] = 0 # Labels of structures at left, top, below
                    # If in bounds look at neighbours
                    if x > 0:
                        neighbour_ids[0] = layer[x-1, y]
                    if y > 0:
                        neighbour_ids[1] = layer[x, y-1]
                    if self.relative_z > 0:
                        neighbour_ids[2] = self.previous_layer[x, y]

                    if self.is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.structure_manager.add(x, y, self.z, neighbour_ids)
                else:
                    struct_id = 0  # reset so that grayscale value does not count as structure in next iterations

                layer[x, y] = struct_id
        return layer

    cdef connect_eight(self, ull[:,:] layer):
        """
        For all the pixels in the current layer, finds all structures touching this pixel using the
        eight connected (connected by edges or corners) rule and also looks at the pixel at the same
        location in the previous layer.
        If structures are found, they are added to the structure manager and the pixel labeled accordingly.

        :param layer:
        :return:
        """
        cdef ull struct_id
        cdef ull neighbour_ids[N_NEIGHBOURS_8_CONNECTED]
        cdef uint i
        for i in range(N_NEIGHBOURS_8_CONNECTED):  # reset
            neighbour_ids[i] = 0  # Labels of neighbour structures touching before

        cdef uint y, x
        for y in range(layer.shape[1]):
            for x in range(layer.shape[0]):
                if layer[x, y] == self.SOMA_CENTRE_VALUE:
                    for i in range(N_NEIGHBOURS_8_CONNECTED):  # reset
                        neighbour_ids[i] = 0

                    # If in bounds look at neighbours
                    if x > 0 and y > 0:
                        neighbour_ids[0] = layer[x-1, y-1]
                    if x > 0:
                        neighbour_ids[1] = layer[x-1, y]
                    if y > 0:
                        neighbour_ids[2] = layer[x, y-1]
                        neighbour_ids[3] = layer[x+1, y-1]
                    if self.relative_z > 0:
                        if x > 0 and y > 0:
                            neighbour_ids[4] = self.previous_layer[x-1, y-1]
                        if x > 0:
                            neighbour_ids[5] = self.previous_layer[x-1, y]
                            if y < layer.shape[1] - 1:
                                neighbour_ids[6] = self.previous_layer[x-1, y+1]
                        if y > 0:
                            neighbour_ids[7] = self.previous_layer[x, y-1]
                            if x < layer.shape[0] - 1:
                                neighbour_ids[8] = self.previous_layer[x+1, y-1]
                        neighbour_ids[9] = self.previous_layer[x, y]
                        if y < layer.shape[1] - 1:
                            neighbour_ids[10] = self.previous_layer[x, y+1]
                        if x < layer.shape[0] - 1:
                            neighbour_ids[11] = self.previous_layer[x+1, y]
                            if y < layer.shape[1] - 1:
                                neighbour_ids[12] = self.previous_layer[x+1, y+1]

                    if self.is_new_structure(neighbour_ids):
                        neighbour_ids[0] = self.next_structure_id
                        self.next_structure_id += 1
                    struct_id = self.structure_manager.add(x, y, self.z, neighbour_ids)
                else:
                    struct_id = 0  # reset so that grayscale value does not count as structure in next iterations

                layer[x, y] = struct_id
        return layer

    @cython.boundscheck(False)
    cdef is_new_structure(self, ull[:] neighbour_ids):  # TEST:
        cdef uint i
        for i in range(len(neighbour_ids)):
            if neighbour_ids[i] != 0:
                return False
        return True

    cpdef get_cell_centres(self):
        cdef CppVector[Point] cell_centres
        cell_centres = self.structure_manager.structures_to_cells()
        return cell_centres

    cpdef get_coords_list(self):
        coords = self.structure_manager.get_coords_dict()  # TODO: cache (attribute)
        return coords

cdef class StructureManager:
    cdef:
        CppMap[ull, ull] obsolete_ids
        CppMap[ull, CppVector[Point]] coords_maps
        int default_cell_type

    def __init__(self):
        self.default_cell_type = Cell.UNKNOWN

    cpdef get_coords_dict(self):
        return self.coords_maps

    @cython.boundscheck(False)
    cdef add(self, uint x, uint y, uint z, ull[:] neighbour_ids):
        """
        For the current coordinates takes all the neighbours and find the minimum structure
        including obsolete structures mapping to any of the neighbours recursively.
        Once the correct structure id is found, append a point with the current coordinates to the coordinates map
        entry for the correct structure. Hence each entry of the map will be a vector of all the pertaining points.

        :param x: 
        :param y: 
        :param z: 
        :param neighbour_ids: 
        :return: 
        """
        cdef ull updated_id

        updated_id = self.sanitise_ids(neighbour_ids)
        self.merge_structures(updated_id, neighbour_ids)

        cdef Point p = Point(x, y, z)  # Necessary to split definition on some machines
        self.coords_maps[updated_id].push_back(p)  # Add point for that structure

        return updated_id

    @cython.boundscheck(False)
    cdef sanitise_ids(self, ull[:] neighbour_ids):
        """
        For all the neighbour ids, walk up the chain of obsolescence (self.obsolete_ids)
        to reassign the corresponding most obsolete structure to the current neighbour

        :param neighbour_ids:
        :return: updated_id
        """
        cdef ull updated_id, neighbour_id
        cdef uint i
        for i in range(len(neighbour_ids)):
            neighbour_id = neighbour_ids[i]
            while self.obsolete_ids.count(neighbour_id):  # walk up the chain of obsolescence
                neighbour_id = self.obsolete_ids[neighbour_id]
            neighbour_ids[i] = neighbour_id

        updated_id = get_non_zero_ull_min(neighbour_ids)  # FIXME: what happens if all neighbour_ids are 0 (raise)
        return updated_id

    @cython.boundscheck(False)
    cdef merge_structures(self, ull updated_id, ull[:] neighbour_ids):
        """
        For all the neighbours, reassign all the points of neighbour to updated_id
        Then deletes the now obsolete entry from the points map and add that entry to the obsolete_ids

        :param updated_id:
        :param neighbour_ids:
        """
        cdef ull neighbour_id
        cdef Point p
        cdef uint i
        for i in range(len(neighbour_ids)):
            neighbour_id = neighbour_ids[i]
            if neighbour_id > updated_id:  # minimise ID so if neighbour with higher ID, reassign its points to current
                for p in self.coords_maps[neighbour_id]:
                    self.coords_maps[updated_id].push_back(p)
                self.coords_maps.erase(neighbour_id)
                self.obsolete_ids[neighbour_id] = updated_id

    cdef structures_to_cells(self):
        cdef CppVector[Point] structure, cell_centres
        cdef Point p

        cdef CppPair[ull, CppVector[Point]] iterator_pair
        for iterator_pair in self.coords_maps:
            structure = iterator_pair.second
            p = get_structure_centre(structure)
            cell_centres.push_back(p)
        return cell_centres
