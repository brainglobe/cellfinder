# cython: language_level=3


cimport libc.math as cmath
cimport cython

import numpy as np
from cellfinder.detect.filters.typedefs cimport uint, ushort

# only for __init__
from cellfinder.tools.array_operations import bin_mean_3d
from cellfinder.tools.geometry import make_sphere

DEBUG = False



cdef class BallFilter:

    cdef:
        uint THRESHOLD_VALUE, SOMA_CENTRE_VALUE
        uint ball_xy_size, tile_step_width, tile_step_height
        int __current_z
        double overlap_fraction, overlap_threshold

        # Numpy arrays
        double[:,:,:] kernel
        ushort[:,:,:] volume
        unsigned char[:,:,:] good_tiles_mask


    def __init__(self, layer_width, layer_height, ball_xy_size, ball_z_size, overlap_fraction=0.8,
                 tile_step_width=None, tile_step_height=None, threshold_value=None, soma_centre_value=None):
        self.ball_xy_size = ball_xy_size
        self.overlap_fraction = overlap_fraction
        self.tile_step_width = tile_step_width
        self.tile_step_height = tile_step_height

        self.THRESHOLD_VALUE = threshold_value
        self.SOMA_CENTRE_VALUE = soma_centre_value

        # temporary kernel of scaling_factor*ball_x_y size to be then scaled to final ball size
        scaling_factor = 2
        x_upscale_factor = y_upscale_factor = z_upscale_factor = 7  # WARNING: needs to be integer
        temp_kernel_shape = [x_upscale_factor * ball_xy_size, y_upscale_factor * ball_xy_size, z_upscale_factor * ball_z_size]
        tmp_ball_centre_position = [cmath.floor(d / 2) for d in temp_kernel_shape]  # z_centre is xy_centre before resize
        tmp_ball_radius = temp_kernel_shape[0] / 2.0
        tmp_kernel = make_sphere(temp_kernel_shape, tmp_ball_radius, tmp_ball_centre_position)
        tmp_kernel = tmp_kernel.astype(np.float64)
        self.kernel = bin_mean_3d(tmp_kernel, x_upscale_factor, y_upscale_factor, z_upscale_factor)

        assert self.kernel.shape[2] == ball_z_size, 'Kernel z dimension should be {}, got {}'\
            .format(ball_z_size, self.kernel.shape[2])

        self.overlap_threshold = self.overlap_fraction * np.array(self.kernel, dtype=np.float64).sum()

        self.volume = np.empty((layer_width, layer_height, self.kernel.shape[2]), dtype=np.uint16)

        self.good_tiles_mask = np.empty((int(cmath.ceil(layer_width / tile_step_width)),  # TODO: lazy initialisation
                                         int(cmath.ceil(layer_height / tile_step_height)),
                                         self.kernel.shape[2]), dtype=np.uint8)
        self.__current_z = -1

    @property
    def ready(self):
        return self.__current_z == self.volume.shape[2] - 1

    cpdef append(self, ushort[:,:] layer, unsigned char[:,:] mask):
        if DEBUG:
            assert [e for e in layer.shape[:2]] == [e for e in self.volume.shape[:2]],\
                'layer shape mismatch, expected "{}", got "{}"'\
                    .format([e for e in self.volume.shape[:2]], [e for e in layer.shape[:2]])
            assert [e for e in mask.shape[:2]] == [e for e in self.good_tiles_mask.shape[2]], \
                'mask shape mismatch, expected"{}", got {}"'\
                    .format([e for e in self.good_tiles_mask.shape[:2]], [e for e in mask.shape[:2]])
        if not self.ready:
            self.__current_z += 1
        else:
            self.volume = np.roll(self.volume, self.volume.shape[2] - 1, axis=2)  # WARNING: not in place
            self.good_tiles_mask = np.roll(self.good_tiles_mask, self.good_tiles_mask.shape[2] - 1, axis=2)
        self.volume[:, :, self.__current_z] = layer[:,:]
        self.good_tiles_mask[:, :, self.__current_z] = mask[:,:]

    cdef get_middle_plane_idx(self):
        cdef uint middle
        middle = <uint> cmath.floor(self.volume.shape[2] / 2)
        return middle

    def get_middle_plane(self):
        cdef uint middle_plane_idx = self.get_middle_plane_idx()
        return np.array(self.volume[:, :, middle_plane_idx], dtype=np.uint16)

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef walk(self):  # Highly optimised because most time critical
        cdef uint stack_middle = <uint> cmath.floor(self.volume.shape[2] / 2)
        cdef uint ball_centre_x, ball_centre_y
        cdef uint ball_radius = self.ball_xy_size // 2
        cdef ushort[:,:,:] cube

        cdef uint max_width, max_height
        tile_mask_covered_img_width = self.good_tiles_mask.shape[0] * self.tile_step_width
        tile_mask_covered_img_height = self.good_tiles_mask.shape[1] * self.tile_step_height
        max_width = tile_mask_covered_img_width - self.ball_xy_size   # whole ball size because the cube is extracted with x + whole ball width
        max_height = tile_mask_covered_img_height - self.ball_xy_size   # whole ball size because the cube is extracted with y + whole ball height
        cdef uint x, y
        for y in range(max_height):
            for x in range(max_width):
                ball_centre_x = x + ball_radius
                ball_centre_y = y + ball_radius
                if self.__is_tile_to_check(ball_centre_x, ball_centre_y):
                    cube = self.volume[x:x + self.kernel.shape[0], y:y + self.kernel.shape[1], :]
                    if self.__cube_overlaps(cube):
                        self.volume[ball_centre_x, ball_centre_y, stack_middle] = self.SOMA_CENTRE_VALUE

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef __cube_overlaps(self, ushort[:,:,:] cube):  # Highly optimised because most time critical
        """

        :param np.ndarray cube: The thresholded array to check for ball fit. values at CellDetector.THRESHOLD_VALUE are threshold
        :return: True if the overlap exceeds self.overlap_fraction
        """
        if DEBUG:
            assert cube.max() <= 1
            assert cube.shape == self.kernel.shape

        cdef double current_overlap_value = 0

        cdef uint x, y, z
        for z in range(cube.shape[2]):  # TODO: OPTIMISE: step from middle to outer boundaries to check more data first
            if z == cmath.floor(self.volume.shape[2] / 2) + 1 and current_overlap_value < self.overlap_threshold * 0.4:  # FIXME: do not hard code value
                return False  # DEBUG: optimisation attempt
            for y in range(cube.shape[1]):
                for x in range(cube.shape[0]):
                    if cube[x, y, z] >= self.THRESHOLD_VALUE:  # includes self.SOMA_CENTRE_VALUE
                        current_overlap_value += self.kernel[x, y, z]
        return <bint> (current_overlap_value > self.overlap_threshold)

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef __is_tile_to_check(self, uint x, uint y):  # Highly optimised because most time critical
        cdef uint x_in_mask, y_in_mask, middle_plane_idx
        x_in_mask = x // self.tile_step_width  # TEST: test bounds (-1 range)
        y_in_mask = y // self.tile_step_height  # TEST: test bounds (-1 range)
        middle_plane_idx = <uint> cmath.floor(self.volume.shape[2] / 2)
        return <bint> self.good_tiles_mask[x_in_mask, y_in_mask, middle_plane_idx]
