# distutils: language = c++

from libcpp.map cimport map as CppMap
from libcpp.vector cimport vector as CppVector
from libcpp.pair cimport pair as CppPair

from cellfinder.detect.filters.typedefs cimport ull, ulong, uint, ushort


cdef get_biggest_structure(CppVector[ulong] sizes):
    cdef uint result = 0
    cdef uint val
    for val in sizes:
        if val > result:
            result = val
    return result


cdef class BaseTileFilter:

    cdef:
        readonly uint out_of_brain_intensity_threshold
        public ushort[:,:] _tile
        public ushort current_threshold
        public bint keep
        SizeAnalyser size_analyser

    def __init__(self, out_of_brain_intensity_threshold=100):
        """

        :param int out_of_brain_intensity_threshold: Set to 0 to disable
        """
        self.out_of_brain_intensity_threshold = out_of_brain_intensity_threshold
        self.current_threshold = -1
        self.keep = True
        self.size_analyser = SizeAnalyser()

    cpdef set_tile(self, tile):
        raise NotImplementedError

    cpdef get_tile(self):
        raise NotImplementedError

    cpdef get_structures(self):
        cdef CppVector[ulong] struct_sizes
        self.size_analyser.process(self._tile, self.current_threshold)
        struct_sizes = self.size_analyser.get_sizes()
        return get_biggest_structure(struct_sizes), struct_sizes.size()

    cpdef is_low_average(self):  # TODO: move to OutOfBrainTileFilter
        cdef bint is_low
        cdef double avg = 0
        cdef uint x, y
        for x in range(self._tile.shape[0]):
            for y in range(self._tile.shape[1]):
                avg += self._tile[x, y]
        avg /= self._tile.shape[0] * self._tile.shape[1]
        is_low = avg < self.out_of_brain_intensity_threshold
        self.keep = not is_low
        return is_low


cdef class OutOfBrainTileFilter(BaseTileFilter):

    cpdef set_tile(self, tile):
        self._tile = tile

    cpdef get_tile(self):
        return self._tile


cdef class SizeAnalyser:

    cdef:
        CppMap[ull, ull] obsolete_ids
        CppMap[ull, ulong] struct_sizes

    cpdef process(self, ushort[:,:] tile, ushort threshold):
        tile = tile.copy()
        self.clear_maps()

        cdef ull struct_id, id_west, id_north, last_structure_id
        last_structure_id = 1

        cdef uint y, x
        for y in range(tile.shape[1]):
            for x in range(tile.shape[0]):
                # default struct_id to 0 so that it is not counted as structure in next iterations
                id_west = id_north = struct_id = 0
                if tile[x, y] >= threshold:
                    # If in bounds look to neighbours
                    if x > 0:
                        id_west = tile[x-1, y]
                    if y > 0:
                        id_north = tile[x, y-1]

                    id_west = self.sanitise_id(id_west)
                    id_north = self.sanitise_id(id_north)

                    if id_west != 0:
                        if id_north != 0 and id_north != id_west:
                            struct_id = self.merge_structures(id_west, id_north)
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

    cpdef get_sizes(self):
        cdef CppVector[ulong] struct_sizes
        cdef ulong size
        cdef ull s_id
        cdef CppPair[ull, ulong] iterator_pair
        for iterator_pair in self.struct_sizes:
            struct_sizes.push_back(iterator_pair.second)
        return struct_sizes

    cdef clear_maps(self):
        self.obsolete_ids.clear()
        self.struct_sizes.clear()

    cdef sanitise_id(self, ull s_id):
        while self.obsolete_ids.count(s_id):  # walk up the chain of obsolescence
            s_id = self.obsolete_ids[s_id]
        return s_id

    # id1 and id2 must be valid struct IDs (>0)!
    cdef merge_structures(self, ull id1, ull id2):
        cdef ulong new_size
        cdef ull tmp_id

        # ensure id1 is the smaller of the two values
        if id2 < id1:
            tmp_id = id1    # swap
            id1 = id2
            id2 = tmp_id

        self.struct_sizes[id1] += self.struct_sizes[id2]
        self.struct_sizes.erase(id2)

        self.obsolete_ids[id2] = id1

        return id1
