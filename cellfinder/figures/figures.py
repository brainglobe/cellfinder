import logging

from cellfinder.figures import heatmap


def run(args, atlas, downsampled_shape):
    logging.info("Generating heatmap")
    heatmap.run(
        args.paths.downsampled_points,
        atlas,
        downsampled_shape,
        args.brainreg_paths.registered_atlas,
        args.paths.heatmap,
        smoothing=args.heatmap_smooth,
        mask=args.mask_figures,
    )
