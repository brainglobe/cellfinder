from imlib.IO.cells import get_cells
import bg_space as bgs
import numpy as np
import imio
import tifffile
import pandas as pd


def export_points(
    points,
    output_directory,
    resolution,
    max_axis_2,
    name="points",
    points_file_extension=".h5",
):
    output_filename = output_directory / (name + points_file_extension)
    points = pd.DataFrame(points * resolution)
    points.columns = ["x", "y", "z"]
    # BR is oriented differently (for now)
    points["z"] = (max_axis_2 * resolution) - points["z"]
    points.to_hdf(output_filename, key="df", mode="w")


def get_brain_structures(
    atlas,
    data_orientation,
    x_pixel_um,
    y_pixel_um,
    z_pixel_um,
    xml_file_path,
    raw_data_path,
    deformation_field_paths,
    output_directory,
):
    cells = get_cells(xml_file_path, cells_only=True)
    cell_list = []
    for cell in cells:
        cell_list.append([cell.z, cell.y, cell.x])
    cells = np.array(cell_list)

    source_shape = tuple(
        imio.get_size_image_from_file_paths(raw_data_path).values()
    )

    source_shape = (source_shape[2], source_shape[1], source_shape[0])

    target_shape = tifffile.imread(deformation_field_paths[0]).shape

    source_space = bgs.SpaceConvention(
        data_orientation,
        shape=source_shape,
        resolution=(z_pixel_um, y_pixel_um, x_pixel_um),
    )
    target_space = bgs.SpaceConvention(
        atlas.metadata["orientation"],
        shape=target_shape,
        resolution=atlas.resolution,
    )
    mapped_points = source_space.map_points_to(target_space, cells)

    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]

    points = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tifffile.imread(deformation_field_path)
        for point in mapped_points:
            point = [int(round(p)) for p in point]
            points[axis].append(
                int(
                    round(
                        field_scales[axis]
                        * deformation_field[point[0], point[1], point[2]]
                    )
                )
            )

    transformed_points = np.array(points).T

    regions = []
    hemispheres = []
    for point in transformed_points:
        try:
            regions.append(atlas.structure_from_coords(point, as_acronym=True))
            hemispheres.append(
                atlas.hemisphere_from_coords(point, as_string=True)
            )
        except:
            continue

    # export to BR
    print("Exporting to brainrender")
    max_axis_2 = atlas.metadata["shape"][2]
    export_points(
        transformed_points,
        output_directory,
        atlas.resolution[0],
        max_axis_2,
        name="cells",
    )

    return regions, hemispheres
