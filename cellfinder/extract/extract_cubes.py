"""
Cube extraction for CNN-based classification.

Based on, and mostly copied from,
https://github.com/SainsburyWellcomeCentre/cell_count_analysis by
Charly Rousseau (https://github.com/crousseau).
"""

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import logging
from datetime import datetime

import numpy as np
from skimage import transform
from numpy.linalg.linalg import LinAlgError
from math import floor
from tifffile import tifffile
from tqdm import tqdm
from imlib.general.system import get_num_processes
from imlib.cells.cells import group_cells_by_z
from imlib.general.numerical import is_even

from cellfinder.tools import image_processing as img_tools
from cellfinder.tools import system


class StackSizeError(Exception):
    pass


class Cube(object):
    class CubeBoundaryError(Exception):
        pass

    def __init__(
        self,
        cell,
        channel,
        stacks,
        x_pix_um=1,
        y_pix_um=1,
        x_pix_um_network=1,
        y_pix_um_network=1,
        final_depth=20,
        width=50,
        height=50,
        depth=20,
        interpolation_order=0,
    ):
        self.cell = cell
        self.channel = channel
        self.stack = stacks[self.channel]
        self.scale_cubes = False
        self.scale_cubes_in_z = False
        self.interpolation_order = interpolation_order
        self.rescaling_factor_x = 1
        self.rescaling_factor_y = 1

        if x_pix_um != x_pix_um_network:
            self.rescaling_factor_x = float(x_pix_um_network) / float(x_pix_um)
            self.scale_cubes = True
        if y_pix_um != y_pix_um_network:
            self.rescaling_factor_y = float(y_pix_um_network) / float(y_pix_um)
            self.scale_cubes = True

        # This should be decided earlier
        if depth != final_depth:
            self.rescaling_factor_z = final_depth / depth
            self.scale_cubes_in_z = True

        self.width = width
        self.height = height
        self.depth = depth
        self.final_depth = depth

        self.empty = False
        self.data = self.extract_cube(self.stack)

    def __str__(self):  # TODO: 0 pad (when changing classification)
        prefix = "pCell"
        return "{}z{}y{}x{}Ch{}.tif".format(
            prefix, self.z, self.y, self.x, self.channel
        )

    @property
    def x(self):
        return self.cell.x

    @property
    def y(self):
        return self.cell.y

    @property
    def z(self):
        return self.cell.z

    @property
    def stack_x_max(self):
        return self.stack.shape[1]

    @property
    def stack_y_max(self):
        return self.stack.shape[0]

    @property
    def stack_z_max(self):
        return self.stack.shape[2]

    def extract_cube(self, stack):
        """

        :param np.array stack:
        :return:
        """
        try:
            self.check_boundaries()
        except Cube.CubeBoundaryError:
            return self.make_blank_data()

        half_width, half_height, x0, x1, y0, y1 = self.get_boundaries()
        data = stack[y0:y1, x0:x1, :]
        if data.ndim < 3:
            logging.error(
                "Error occurred during extraction of cube {}, "
                "original shape: {}, dtype: {}".format(
                    self, data.shape, data.dtype
                )
            )
            return self.make_blank_data()

        out = []
        # TODO: combine these scalings
        for i in range(self.depth):
            if self.scale_cubes:
                try:
                    # To match classification 1um rescaling
                    inv_rescale_x = 1 / self.rescaling_factor_x
                    inv_rescale_y = 1 / self.rescaling_factor_y
                    plane = transform.rescale(
                        data[:, :, i],
                        scale=(inv_rescale_y, inv_rescale_x),
                        order=self.interpolation_order,
                        preserve_range=True,
                        multichannel=False,
                        mode="constant",
                        anti_aliasing=False,
                    )
                except LinAlgError as err:
                    logging.error(
                        f"Error occurred during rescale of cube {self}, "
                        f"original shape: {data[:, :, i].shape}, dtype: "
                        f"{data[:, :, i].dtype}; {err}."
                    )

                    return self.make_blank_data()
            else:
                plane = data[:, :, i]

            plane = self.adjust_size(plane)

            out.append(plane)
        out = np.array(out, dtype=np.uint16)

        if self.scale_cubes_in_z:
            try:
                # To match classification cube scaling in z
                # TODO: presumably this will have rounding errors
                out = transform.rescale(
                    out,
                    scale=(self.rescaling_factor_z, 1, 1),
                    order=self.interpolation_order,
                    preserve_range=True,
                    multichannel=False,
                    mode="constant",
                    anti_aliasing=False,
                )
                out = out.astype(np.uint16)
            except LinAlgError as err:
                logging.error(
                    f"Error occurred during z-rescale of cube {self}, "
                    f"from depth: {self.depth} to final depth  "
                    f"{self.final_depth} ; {err}"
                )
                print.error(
                    f"Error occurred during z-rescale of cube {self}, "
                    f"from depth: {self.depth} to final depth  "
                    f"{self.final_depth} ; {err}"
                )
                return self.make_blank_data()
        return out

    def adjust_size(self, plane):
        plane = self.crop_to_correct_size(plane)
        plane = self.pad_to_correct_size(plane)
        return plane

    def pad_to_correct_size(self, plane):
        return img_tools.pad_center_2d(
            plane, x_size=self.width, y_size=self.height
        )

    def crop_to_correct_size(self, plane):
        return img_tools.crop_center_2d(
            plane, crop_x=self.width, crop_y=self.height
        )

    def get_boundaries(self):
        if self.scale_cubes:
            half_width = int(round(self.width / 2 * self.rescaling_factor_x))
            half_height = int(round(self.height / 2 * self.rescaling_factor_y))
        else:
            half_width = int(round(self.width / 2))
            half_height = int(round(self.height / 2))

        x0 = int(self.x - half_width)
        x1 = int(self.x + half_width)
        y0 = int(self.y - half_height)
        y1 = int(self.y + half_height)
        return half_width, half_height, x0, x1, y0, y1

    def make_blank_data(self):
        self.empty = True
        return np.zeros((self.depth, self.width, self.height), dtype=np.uint16)

    def check_boundaries(self):
        out_bounds_message = (
            "Cell candidate (x:{}, y:{}, z:{}) too close to the edge of the"
            " image for cube extraction."
        ).format(self.x, self.y, self.z)

        if self.stack_z_max < self.depth:
            error_msg = "Stack has {} planes but cube needs {}".format(
                self.stack_z_max, self.depth
            )
            logging.error(error_msg)

            raise Cube.CubeBoundaryError(error_msg)
        boundaries = self.get_boundaries()
        half_height, half_width, x0, x1, y0, y1 = boundaries
        if x0 < 0 or y0 < 0:
            logging.warning(out_bounds_message)
            raise Cube.CubeBoundaryError(out_bounds_message)

        # WARNING: dimensions inverted (numpy y=dim0, x=dim1)
        elif x1 > self.stack_x_max or y1 > self.stack_y_max:
            logging.warning(out_bounds_message)
            raise Cube.CubeBoundaryError(out_bounds_message)


def save_cubes(
    cells,
    planes_paths,
    planes_to_read,
    planes_shape,
    voxel_sizes,
    network_voxel_sizes,
    num_planes_for_cube=20,
    cube_width=50,
    cube_height=50,
    cube_depth=20,
    thread_id=0,
    output_dir="",
    save_empty_cubes=False,
):
    """

    :param cells:
    :param planes_paths:
    :param planes_to_read:
    :param planes_shape:
    :param x_pix_um:
    :param y_pix_um:
    :param x_pix_um_network:
    :param y_pix_um_network:
    :param num_planes_for_cube:
    :param cube_width:
    :param cube_height:
    :param cube_depth:
    :param thread_id:
    :param output_dir:
    :param save_empty_cubes:
    :return:
    """
    channels = list(planes_paths.keys())
    stack_shape = planes_shape + (num_planes_for_cube,)
    stacks = {}
    planes_queues = {}
    for ch in channels:
        stacks[ch] = np.zeros(stack_shape, dtype=np.uint16)
        planes_queues[ch] = deque(maxlen=num_planes_for_cube)
    for plane_idx in tqdm(planes_to_read, desc="Thread: {}".format(thread_id)):
        for ch in channels:
            plane_path = planes_paths[ch][plane_idx]
            planes_queues[ch].append(tifffile.imread(plane_path))
            if len(planes_queues[ch]) == num_planes_for_cube:
                if is_even(num_planes_for_cube):
                    cell_z = int(plane_idx - num_planes_for_cube / 2 + 1)
                else:
                    cell_z = int(
                        plane_idx - floor(num_planes_for_cube) / 2 + 1
                    )

                for j, plane in enumerate(planes_queues[ch]):
                    stacks[ch][:, :, j] = plane

                # ensures no cube_depth planes at the end
                planes_queues[ch].popleft()
                # required since we provide all cells
                # TODO: if len(planes_queues[ch])
                #  < num_planes_for_cube -1: break
                for cell in cells[cell_z]:
                    cube = Cube(
                        cell,
                        ch,
                        stacks,
                        x_pix_um=voxel_sizes[2],
                        y_pix_um=voxel_sizes[1],
                        x_pix_um_network=network_voxel_sizes[2],
                        y_pix_um_network=network_voxel_sizes[1],
                        final_depth=cube_depth,
                        width=cube_width,
                        height=cube_height,
                        depth=num_planes_for_cube,
                    )
                    if not cube.empty or (cube.empty and save_empty_cubes):
                        tifffile.imsave(
                            os.path.join(output_dir, str(cube)), cube.data
                        )


def get_ram_requirement_per_process(
    single_image_path, cube_depth, num_channels=2, copies=2
):
    """
    Calculates how much RAM is needed per CPU core for cube extraction. Used
    later to set the number of CPU cores to be used.

    RAM requirement is currently:
    image_size x n_cores x n_channels x cube_depth x copies

    :param single_image_path: A single 2D image, used to find the size.
    :param num_channels: Number of channels to be extracted
    :param cube_depth: Depth of the cube to be extracted (z-planes)
    to use.
    :param int copies: How many copies of the data are loaded at any time
    :return int: memory requirement for one CPU core
    """

    logging.debug(
        "Determining how much RAM is needed for each CPU for cube extraction"
    )
    file_size = os.path.getsize(single_image_path)
    total_mem_need = file_size * num_channels * cube_depth * copies

    logging.debug(
        "File size: {:.2f} GB, "
        "number of channels: {}, "
        "cube depth: {}, "
        "Therefore theoretically {:.2f} GB is needed per CPU core."
        "".format(
            (file_size / (1024 ** 3)),
            num_channels,
            cube_depth,
            (total_mem_need / (1024 ** 3)),
        )
    )

    return total_mem_need


def main(
    cells,
    cubes_output_dir,
    planes_paths,
    cube_depth,
    cube_width,
    cube_height,
    voxel_sizes,
    network_voxel_sizes,
    max_ram,
    n_free_cpus=4,
    save_empty_cubes=False,
):

    start_time = datetime.now()

    if voxel_sizes[0] != network_voxel_sizes[0]:
        plane_scaling_factor = float(network_voxel_sizes[0]) / float(
            voxel_sizes[0]
        )
        num_planes_needed_for_cube = round(cube_depth * plane_scaling_factor)
    else:
        num_planes_needed_for_cube = cube_depth

    if num_planes_needed_for_cube > len(planes_paths[0]):
        raise StackSizeError(
            "The number of planes provided is not sufficient "
            "for any cubes to be extracted. Please check the "
            "input data"
        )

    first_plane = tifffile.imread(list(planes_paths.values())[0][0])

    planes_shape = first_plane.shape
    brain_depth = len(list(planes_paths.values())[0])

    # TODO: use to assert all centre planes processed
    center_planes = sorted(list(set([cell.z for cell in cells])))

    # REFACTOR: rename (clashes with different meaning of planes_to_read below)
    planes_to_read = np.zeros(brain_depth, dtype=np.bool)

    if is_even(num_planes_needed_for_cube):
        half_nz = num_planes_needed_for_cube // 2
        # WARNING: not centered because even
        for p in center_planes:
            planes_to_read[p - half_nz : p + half_nz] = 1
    else:
        half_nz = num_planes_needed_for_cube // 2
        # centered
        for p in center_planes:
            planes_to_read[p - half_nz : p + half_nz + 1] = 1

    planes_to_read = np.where(planes_to_read)[0]

    if not planes_to_read.size:
        logging.error(
            f"No planes found, you need at the very least "
            f"{num_planes_needed_for_cube} "
            f"planes to proceed (i.e. cube z size)"
            f"Brain z dimension is {brain_depth}.",
            stack_info=True,
        )
        raise ValueError(
            f"No planes found, you need at the very least "
            f"{num_planes_needed_for_cube} "
            f"planes to proceed (i.e. cube z size)"
            f"Brain z dimension is {brain_depth}."
        )
    # TODO: check if needs to flip args.cube_width and args.cube_height
    cells_groups = group_cells_by_z(cells)

    # copies=2 is set because at all times there is a plane queue (deque)
    # and an array passed to `Cube`
    ram_per_process = get_ram_requirement_per_process(
        planes_paths[0][0],
        num_planes_needed_for_cube,
        copies=2,
    )
    n_processes = get_num_processes(
        min_free_cpu_cores=n_free_cpus,
        ram_needed_per_process=ram_per_process,
        n_max_processes=len(planes_to_read),
        fraction_free_ram=0.2,
        max_ram_usage=system.memory_in_bytes(max_ram, "GB"),
    )
    # TODO: don't need to extract cubes from all channels if
    #  n_signal_channels>1
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        n_planes_per_chunk = len(planes_to_read) // n_processes
        for i in range(n_processes):
            start_idx = i * n_planes_per_chunk
            end_idx = (
                start_idx + n_planes_per_chunk + num_planes_needed_for_cube - 1
            )
            if end_idx > planes_to_read[-1]:
                end_idx = None
            sub_planes_to_read = planes_to_read[start_idx:end_idx]

            executor.submit(
                save_cubes,
                cells_groups,
                planes_paths,
                sub_planes_to_read,
                planes_shape,
                voxel_sizes,
                network_voxel_sizes,
                num_planes_for_cube=num_planes_needed_for_cube,
                cube_width=cube_width,
                cube_height=cube_height,
                cube_depth=cube_depth,
                thread_id=i,
                output_dir=cubes_output_dir,
                save_empty_cubes=save_empty_cubes,
            )

    total_cubes = system.get_number_of_files_in_dir(cubes_output_dir)
    time_taken = datetime.now() - start_time
    logging.info(
        "All cubes ({}) extracted in: {}".format(total_cubes, time_taken)
    )
