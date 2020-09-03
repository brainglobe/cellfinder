from imlib.IO.cells import get_cells
import bg_space as bgs
import numpy as np
import imio
import tifffile
import pandas as pd
import os
from imlib.pandas.misc import sanitise_df


def export_points(
    points,
    atlas,
    output_directory,
    resolution,
    name="points",
    points_file_extension=".h5",
):
    print("Exporting to brainrender")
    max_axis_2 = atlas.metadata["shape"][2]
    output_filename = output_directory / (name + points_file_extension)
    points = pd.DataFrame(points * resolution)
    points.columns = ["x", "y", "z"]
    # BR is oriented differently (for now)
    points["z"] = (max_axis_2 * resolution) - points["z"]
    points.to_hdf(output_filename, key="df", mode="w")


class Point:
    def __init__(self, structure, hemisphere):
        self.structure = structure
        self.hemisphere = hemisphere


def calculate_densities(counts, volume_csv_path):
    """
    Use the region volume information from registration to calculate cell
    densities. Based on the atlas names, which must be exactly equal.
    :param counts: dataframe with cell counts
    :param volume_csv_path: path of the volumes of each brain region
    :return:
    """
    volumes = pd.read_csv(volume_csv_path, sep=",", header=0, quotechar='"')
    df = pd.merge(counts, volumes, on="structure_name", how="outer")
    df = df.fillna(0)
    df["left_cells_per_mm3"] = df.left_cell_count / df.left_volume_mm3
    df["right_cells_per_mm3"] = df.right_cell_count / df.right_volume_mm3
    return df


def combine_df_hemispheres(df):
    """
    Combine left and right hemisphere data onto a single row
    :param df:
    :return:
    """
    left = df[df["hemisphere"] == "left"]
    right = df[df["hemisphere"] == "right"]
    left = left.drop(["hemisphere"], axis=1)
    right = right.drop(["hemisphere"], axis=1)
    left.rename(columns={"cell_count": "left_cell_count"}, inplace=True)
    right.rename(columns={"cell_count": "right_cell_count"}, inplace=True)
    both = pd.merge(left, right, on="structure_name", how="outer")
    both = both.fillna(0)
    both["total_cells"] = both.left_cell_count + both.right_cell_count
    both = both.sort_values("total_cells", ascending=False)
    return both


def summarise_points(
    transformed_points,
    atlas,
    volume_csv_path,
    output_dir,
    file_name="summary_cell_counts.csv",
):
    points = []
    structures_with_points = set()
    for point in transformed_points:
        try:
            structure = atlas.structure_from_coords(point)
            structure = atlas.structures[structure]["name"]
            hemisphere = atlas.hemisphere_from_coords(point, as_string=True)
            points.append(Point(structure, hemisphere))
            structures_with_points.add(structure)
        except:
            continue

    structures_with_points = list(structures_with_points)
    point_numbers = pd.DataFrame(
        columns=("structure_name", "hemisphere", "cell_count")
    )
    for structure in structures_with_points:
        for hemisphere in ("left", "right"):
            n_points = len(
                [
                    point
                    for point in points
                    if point.structure == structure
                    and point.hemisphere == hemisphere
                ]
            )
            if n_points:
                point_numbers = point_numbers.append(
                    {
                        "structure_name": structure,
                        "hemisphere": hemisphere,
                        "cell_count": n_points,
                    },
                    ignore_index=True,
                )
    sorted_point_numbers = point_numbers.sort_values(
        by=["cell_count"], ascending=False
    )

    combined_hemispheres = combine_df_hemispheres(sorted_point_numbers)
    df = calculate_densities(combined_hemispheres, volume_csv_path)
    df = sanitise_df(df)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, file_name)
    df.to_csv(output_file, index=False)

    return df


def transform_cells_to_atlas_space(
    cells, source_space, atlas, deformation_field_paths,
):
    target_shape = tifffile.imread(deformation_field_paths[0]).shape

    target_space = bgs.AnatomicalSpace(
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
    return transformed_points


def get_brain_structures(
    atlas,
    data_orientation,
    x_pixel_um,
    y_pixel_um,
    z_pixel_um,
    xml_file_path,
    raw_data_path,
    deformation_field_paths,
    volume_csv_path,
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

    source_space = bgs.AnatomicalSpace(
        data_orientation,
        shape=source_shape,
        resolution=(z_pixel_um, y_pixel_um, x_pixel_um),
    )

    transformed_cells = transform_cells_to_atlas_space(
        cells, source_space, atlas, deformation_field_paths
    )

    summarise_points(
        transformed_cells, atlas, volume_csv_path, output_directory
    )

    export_points(
        transformed_cells,
        atlas,
        output_directory,
        atlas.resolution[0],
        name="cells",
    )

    # return structures, hemispheres, structures_with_cells
