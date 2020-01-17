import logging

from cellfinder.figures import heatmap
from amap.config.atlas import Atlas
from cellfinder.tools.source_files import source_custom_config
import cellfinder.tools.brain as brain_tools

from brainio import brainio


def figures(args):
    scales = GetScales(args)
    if args.heatmap:
        logging.info("Generating heatmap")
        heatmap.heatmap(
            args,
            scales.figure_image_shape,
            scales.raw_image_shape,
            scales.bin_size_raw_voxels,
            smoothing=scales.smoothing_target_voxel,
            mask=args.mask_figures,
            atlas=scales.atlas,
            atlas_scale=scales.atlas_scale,
            transformation_matrix=scales.transformation_matrix,
        )


class GetScales:
    # N.B this is all in x,y,z FIJI space, and may need to converted to
    # numpy coordinates
    def __init__(self, args, figure_target="atlas", rounding_decimals=5):
        logging.info("Determining image size for figures.")

        if figure_target is "atlas":
            # N.B assumes an isotropic atlas as target
            self._figure_target_image = args.paths.downsampled_background
        else:
            raise NotImplementedError(
                "Currently the output figures must be of the resolution of "
                "the atlas. These can be changed manually (e.g. in FIJI), "
                "and will eventually be implemented in cellfinder."
            )
        self.scaling_rounding_decimals = rounding_decimals

        self.x_pixel_um = args.x_pixel_um
        self.y_pixel_um = args.y_pixel_um
        self.z_pixel_um = args.z_pixel_um

        self._registration_config = args.registration_config
        self._input_image = args.background_planes_path[0]

        self._bin_um = args.heatmap_binning
        self._smooth_um = args.heatmap_smooth

        self._atlas_path = args.paths.registered_atlas_path
        self._atlas_config = args.atlas_config

        self.atlas_pixel_x_um = None
        self.atlas_pixel_y_um = None
        self.atlas_pixel_z_um = None

        self.x_scaling = None
        self.y_scaling = None
        self.z_scaling = None
        self.position_scaling = [None, None, None]

        self.raw_image_shape = None
        self.figure_image_shape = None
        self.bin_target_voxels = None
        self.bin_size_raw_voxels = None

        self.smoothing_target_voxel = None
        self.atlas = None
        self.atlas_scale = None
        self.transformation_matrix = None

        self.get_registration_config()
        self.get_atlas_config()
        self.get_scaling()
        self.get_raw_image_shape()
        self.get_figure_image_shape()

        self.get_binning()
        self.get_smoothing()

        if args.mask_figures or args.outlines:
            self.get_atlas()

    def get_registration_config(self):
        if self._registration_config is None:
            self._registration_config = source_custom_config()

    def get_atlas_config(self):
        if self._atlas_config is None:
            self._atlas_config = source_custom_config()

    def get_scaling(self):
        logging.debug("Determining scaling of the figures to the raw image")
        atlas = Atlas(self._registration_config)
        atlas_pixel_sizes = atlas.pix_sizes

        self.atlas_pixel_x_um = atlas_pixel_sizes["x"]
        self.atlas_pixel_y_um = atlas_pixel_sizes["y"]
        self.atlas_pixel_z_um = atlas_pixel_sizes["z"]

        self.x_scaling = round(
            self.x_pixel_um / self.atlas_pixel_x_um,
            self.scaling_rounding_decimals,
        )
        self.y_scaling = round(
            self.y_pixel_um / self.atlas_pixel_y_um,
            self.scaling_rounding_decimals,
        )
        self.z_scaling = round(
            self.z_pixel_um / self.atlas_pixel_z_um,
            self.scaling_rounding_decimals,
        )
        self.position_scaling = [
            self.x_scaling,
            self.y_scaling,
            self.z_scaling,
        ]

    def get_raw_image_shape(self):
        logging.debug("Checking raw image size")
        self.raw_image_shape = brainio.get_size_image_from_file_paths(
            self._input_image
        )
        logging.debug(f"Raw image size: {self.raw_image_shape}")

    def get_figure_image_shape(self):
        logging.debug(
            "Loading file: {} to check target image size"
            "".format(self._figure_target_image)
        )
        downsampled_image = brainio.load_nii(self._figure_target_image)
        shape = downsampled_image.shape
        self.figure_image_shape = {"x": shape[0], "y": shape[1], "z": shape[2]}
        logging.debug("Target image size: {}".format(self.figure_image_shape))

    def get_binning(self):
        logging.debug("Calculating bin size in raw image space voxels")
        bin_raw_x = int(self._bin_um / self.x_pixel_um)
        bin_raw_y = int(self._bin_um / self.y_pixel_um)
        bin_raw_z = int(self._bin_um / self.z_pixel_um)
        self.bin_size_raw_voxels = [bin_raw_x, bin_raw_y, bin_raw_z]
        logging.debug(
            f"Bin size in raw image space is x:{bin_raw_x}, "
            f"y:{bin_raw_y}, z:{bin_raw_z}."
        )

    def get_smoothing(self):
        logging.debug(
            "Calculating smoothing in target image volume. Assumes "
            "an isotropic target image"
        )
        if self._smooth_um is not 0:
            self.smoothing_target_voxel = int(
                self._smooth_um / self.atlas_pixel_x_um
            )

    def get_atlas(self):
        atlas = brainio.load_nii(self._atlas_path, as_array=False)
        self.atlas_scale = atlas.header.get_zooms()
        self.atlas = atlas.get_data()
        self.get_transformation_matrix()

    def get_transformation_matrix(self):
        self.transformation_matrix = brain_tools.get_transformation_matrix(
            self._atlas_config
        )
