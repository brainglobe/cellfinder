import numpy as np
from numba import jit

from cellfinder_core.tools.array_operations import bin_mean_3d
from cellfinder_core.tools.geometry import make_sphere

DEBUG = False


class BallFilter:
    def __init__(
        self,
        layer_width,
        layer_height,
        ball_xy_size,
        ball_z_size,
        overlap_fraction=0.8,
        tile_step_width=None,
        tile_step_height=None,
        threshold_value=None,
        soma_centre_value=None,
    ):
        self.ball_xy_size = ball_xy_size
        self.ball_z_size = ball_z_size
        self.overlap_fraction = overlap_fraction
        self.tile_step_width = tile_step_width
        self.tile_step_height = tile_step_height

        self.THRESHOLD_VALUE = threshold_value
        self.SOMA_CENTRE_VALUE = soma_centre_value

        # temporary kernel of scaling_factor*ball_x_y size to be then scaled
        # to final ball size
        x_upscale_factor = (
            y_upscale_factor
        ) = z_upscale_factor = 7  # WARNING: needs to be integer
        temp_kernel_shape = [
            x_upscale_factor * ball_xy_size,
            y_upscale_factor * ball_xy_size,
            z_upscale_factor * ball_z_size,
        ]
        tmp_ball_centre_position = [
            np.floor(d / 2) for d in temp_kernel_shape
        ]  # z_centre is xy_centre before resize
        tmp_ball_radius = temp_kernel_shape[0] / 2.0
        tmp_kernel = make_sphere(
            temp_kernel_shape, tmp_ball_radius, tmp_ball_centre_position
        )
        tmp_kernel = tmp_kernel.astype(np.float64)
        self.kernel = bin_mean_3d(
            tmp_kernel, x_upscale_factor, y_upscale_factor, z_upscale_factor
        )

        assert (
            self.kernel.shape[2] == ball_z_size
        ), "Kernel z dimension should be {}, got {}".format(
            ball_z_size, self.kernel.shape[2]
        )

        self.overlap_threshold = (
            self.overlap_fraction
            * np.array(self.kernel, dtype=np.float64).sum()
        )

        # Stores the current planes that are being filtered
        self.volume = np.empty(
            (layer_width, layer_height, ball_z_size), dtype=np.uint16
        )
        # Index of the middle plane in the volume
        self.middle_z_idx = int(np.floor(ball_z_size / 2))

        self.good_tiles_mask = np.empty(
            (
                int(
                    np.ceil(layer_width / tile_step_width)
                ),  # TODO: lazy initialisation
                int(np.ceil(layer_height / tile_step_height)),
                ball_z_size,
            ),
            dtype=np.uint8,
        )
        # Stores the z-index in volume at which new layers are inserted when
        # append() is called
        self.__current_z = -1

    @property
    def ready(self):
        """
        Return `True` if enough layers have been appended to run the filter.
        """
        return self.__current_z == self.ball_z_size - 1

    def append(self, layer, mask):
        """
        Add a new 2D layer to the filter.
        """
        if DEBUG:
            assert [e for e in layer.shape[:2]] == [
                e for e in self.volume.shape[:2]
            ], 'layer shape mismatch, expected "{}", got "{}"'.format(
                [e for e in self.volume.shape[:2]],
                [e for e in layer.shape[:2]],
            )
            assert [e for e in mask.shape[:2]] == [
                e for e in self.good_tiles_mask.shape[2]
            ], 'mask shape mismatch, expected"{}", got {}"'.format(
                [e for e in self.good_tiles_mask.shape[:2]],
                [e for e in mask.shape[:2]],
            )
        if not self.ready:
            self.__current_z += 1
        else:
            # Shift everything down by one to make way for the new layer
            self.volume = np.roll(
                self.volume, -1, axis=2
            )  # WARNING: not in place
            self.good_tiles_mask = np.roll(self.good_tiles_mask, -1, axis=2)
        # Add the new layer to the top of volume and good_tiles_mask
        self.volume[:, :, self.__current_z] = layer[:, :]
        self.good_tiles_mask[:, :, self.__current_z] = mask[:, :]

    def get_middle_plane(self):
        """
        Get the plane in the middle of self.volume.
        """
        z = self.middle_z_idx
        return np.array(self.volume[:, :, z], dtype=np.uint16)

    def walk(self):  # Highly optimised because most time critical
        ball_radius = self.ball_xy_size // 2
        middle_z = self.middle_z_idx

        tile_mask_covered_img_width = (
            self.good_tiles_mask.shape[0] * self.tile_step_width
        )
        tile_mask_covered_img_height = (
            self.good_tiles_mask.shape[1] * self.tile_step_height
        )
        # whole ball size because the cube is extracted with x + whole ball
        # width
        max_width = tile_mask_covered_img_width - self.ball_xy_size
        # whole ball size because the cube is extracted with y + whole ball
        # height
        max_height = tile_mask_covered_img_height - self.ball_xy_size
        for y in range(max_height):
            for x in range(max_width):
                ball_centre_x = x + ball_radius
                ball_centre_y = y + ball_radius
                if self.__is_tile_to_check(ball_centre_x, ball_centre_y):
                    cube = self.volume[
                        x : x + self.kernel.shape[0],
                        y : y + self.kernel.shape[1],
                        :,
                    ]
                    if _cube_overlaps(
                        cube,
                        self.ball_z_size,
                        self.overlap_threshold,
                        self.THRESHOLD_VALUE,
                        self.kernel,
                    ):
                        self.volume[
                            ball_centre_x, ball_centre_y, middle_z
                        ] = self.SOMA_CENTRE_VALUE

    def __is_tile_to_check(
        self, x, y
    ):  # Highly optimised because most time critical
        middle_z = self.middle_z_idx

        x_in_mask = x // self.tile_step_width  # TEST: test bounds (-1 range)
        y_in_mask = y // self.tile_step_height  # TEST: test bounds (-1 range)
        return self.good_tiles_mask[x_in_mask, y_in_mask, middle_z]


@jit(nopython=True, cache=True)
def _cube_overlaps(
    cube, ball_z_size, overlap_threshold, THRESHOLD_VALUE, kernel
):  # Highly optimised because most time critical
    """

    :param np.ndarray cube: The thresholded array to check for ball fit.
        values at CellDetector.THRESHOLD_VALUE are threshold
    :return: True if the overlap exceeds self.overlap_fraction
    """
    current_overlap_value = 0

    middle = np.floor(ball_z_size / 2) + 1
    overlap_thresh = overlap_threshold * 0.4

    for z in range(cube.shape[2]):
        # TODO: OPTIMISE: step from middle to outer boundaries to check
        # more data first
        # FIXME: do not hard code value
        if z == middle and current_overlap_value < overlap_thresh:
            return False  # DEBUG: optimisation attempt
        for y in range(cube.shape[1]):
            for x in range(cube.shape[0]):
                # includes self.SOMA_CENTRE_VALUE
                if cube[x, y, z] >= THRESHOLD_VALUE:
                    current_overlap_value += kernel[x, y, z]
    return current_overlap_value > overlap_threshold
