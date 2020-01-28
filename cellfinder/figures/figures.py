import logging

from neuro.heatmap.heatmap import main as heatmap


def figures(args):
    if args.heatmap:
        logging.info("Generating heatmap")
        heatmap(
            args.paths.classification_out_file,
            args.paths.heatmap,
            args.background_planes_path[0],
            args.paths.registered_atlas_path,
            args.heatmap_binning,
            args.x_pixel_um,
            args.y_pixel_um,
            args.z_pixel_um,
            args.heatmap_smooth,
            args.mask_figures,
        )
