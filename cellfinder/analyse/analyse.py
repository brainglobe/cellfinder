"""
Cell position analysis (based on atlas registration).

Based on https://github.com/SainsburyWellcomeCentre/cell_count_analysis by
Charly Rousseau (https://github.com/crousseau).
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import bg_space as bgs
import imio
import numpy as np
import pandas as pd
import tifffile
from bg_atlasapi import BrainGlobeAtlas
from brainglobe_utils.general.system import ensure_directory_exists
from brainglobe_utils.pandas.misc import safe_pandas_concat, sanitise_df

from cellfinder.export.export import export_points


class Point:
    def __init__(
        self,
        raw_coordinate,
        atlas_coordinate,
        structure,
        structure_id,
        hemisphere,
    ):
        self.raw_coordinate = raw_coordinate
        self.atlas_coordinate = atlas_coordinate
        self.structure = structure
        self.structure_id = structure_id
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
    raw_points,
    transformed_points,
    atlas,
    volume_csv_path,
    all_points_filename,
    summary_filename,
):
    points = []
    structures_with_points = set()
    for idx, point in enumerate(transformed_points):
        try:
            structure_id = atlas.structure_from_coords(point)
            structure = atlas.structures[structure_id]["name"]
            hemisphere = atlas.hemisphere_from_coords(point, as_string=True)
            points.append(
                Point(
                    raw_points[idx], point, structure, structure_id, hemisphere
                )
            )
            structures_with_points.add(structure)
        except Exception:
            continue

    logging.debug("Ensuring output directory exists")
    ensure_directory_exists(Path(all_points_filename).parent)
    create_all_cell_csv(points, all_points_filename)

    get_region_totals(
        points, structures_with_points, volume_csv_path, summary_filename
    )

    return points


def create_all_cell_csv(points, all_points_filename):
    df = pd.DataFrame(
        columns=(
            "coordinate_raw_axis_0",
            "coordinate_raw_axis_1",
            "coordinate_raw_axis_2",
            "coordinate_atlas_axis_0",
            "coordinate_atlas_axis_1",
            "coordinate_atlas_axis_2",
            "structure_name",
            "hemisphere",
        )
    )

    temp_matrix = [[] for i in range(len(points))]
    for i, point in enumerate(points):
        temp_matrix[i].append(point.raw_coordinate[0])
        temp_matrix[i].append(point.raw_coordinate[1])
        temp_matrix[i].append(point.raw_coordinate[2])
        temp_matrix[i].append(point.atlas_coordinate[0])
        temp_matrix[i].append(point.atlas_coordinate[1])
        temp_matrix[i].append(point.atlas_coordinate[2])
        temp_matrix[i].append(point.structure)
        temp_matrix[i].append(point.hemisphere)

    df = pd.DataFrame(temp_matrix, columns=df.columns, index=None)
    df.to_csv(all_points_filename, index=False)


def get_region_totals(
    points, structures_with_points, volume_csv_path, output_filename
):
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
                point_numbers = safe_pandas_concat(
                    point_numbers,
                    pd.DataFrame(
                        data=[[structure, hemisphere, n_points]],
                        columns=[
                            "structure_name",
                            "hemisphere",
                            "cell_count",
                        ],
                    ),
                )
    sorted_point_numbers = point_numbers.sort_values(
        by=["cell_count"], ascending=False
    )

    combined_hemispheres = combine_df_hemispheres(sorted_point_numbers)
    df = calculate_densities(combined_hemispheres, volume_csv_path)
    df = sanitise_df(df)

    df.to_csv(output_filename, index=False)


def transform_points_to_downsampled_space(
    points: np.ndarray,
    target_space: bgs.AnatomicalSpace,
    source_space: bgs.AnatomicalSpace,
    output_filename: Optional[os.PathLike] = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    points :
        Points in the original space.
    target_space :
        Target anatomical space.
    source_space :
        Source anatomical space of ``points``.
    output_filename :
        File to save downsampled points to. Points are saved as a HDF file
        using pandas.

    Returns
    -------
    downsampled_points
        Points transformed to the downsampled space.
    """
    points = source_space.map_points_to(target_space, points)
    if output_filename is not None:
        df = pd.DataFrame(points)
        df.to_hdf(output_filename, key="df", mode="w")
    return points


def transform_points_to_atlas_space(
    points: np.ndarray,
    source_space: bgs.AnatomicalSpace,
    atlas: BrainGlobeAtlas,
    deformation_field_paths: List[os.PathLike],
    downsampled_space: bgs.AnatomicalSpace,
    downsampled_points_path: Optional[os.PathLike] = None,
    atlas_points_path: Optional[os.PathLike] = None,
) -> np.ndarray:
    """
    Transform points to an atlas space.

    The points are first downsampled to the atlas resolution, and then
    transformed to the atlas space.
    """
    downsampled_points = transform_points_to_downsampled_space(
        points,
        downsampled_space,
        source_space,
        output_filename=downsampled_points_path,
    )
    return transform_points_downsampled_to_atlas_space(
        downsampled_points,
        atlas,
        deformation_field_paths,
        output_filename=atlas_points_path,
    )


def transform_points_downsampled_to_atlas_space(
    downsampled_points: np.ndarray,
    atlas: BrainGlobeAtlas,
    deformation_field_paths: List[os.PathLike],
    output_filename: Optional[os.PathLike] = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    downsampled_points :
        Points already downsampled to the atlas resolution.
    atlas :
        Target atlas.
    deformation_field_paths :
        File paths to the deformation fields.
    output_filename :
        File to save transformed points to. Points are saved as a HDF file
        using pandas.

    Returns
    -------
    transformed_points
        Points transformed to the atlas space.
    """
    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
    points: List[List] = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tifffile.imread(deformation_field_path)
        for point in downsampled_points:
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

    if output_filename is not None:
        df = pd.DataFrame(transformed_points)
        df.to_hdf(output_filename, key="df", mode="w")

    return transformed_points


def run(args, cells, atlas, downsampled_space):
    deformation_field_paths = [
        args.brainreg_paths.deformation_field_0,
        args.brainreg_paths.deformation_field_1,
        args.brainreg_paths.deformation_field_2,
    ]

    cell_list = []
    for cell in cells:
        cell_list.append([cell.z, cell.y, cell.x])
    cells = np.array(cell_list)

    run_analysis(
        cells,
        args.signal_planes_paths[0],
        args.orientation,
        args.voxel_sizes,
        atlas,
        deformation_field_paths,
        downsampled_space,
        args.paths.downsampled_points,
        args.paths.atlas_points,
        args.paths.brainrender_points,
        args.paths.abc4d_points,
        args.brainreg_paths.volume_csv_path,
        args.paths.all_points_csv,
        args.paths.summary_csv,
    )


def run_analysis(
    cells,
    signal_planes,
    orientation,
    voxel_sizes,
    atlas,
    deformation_field_paths,
    downsampled_space,
    downsampled_points_path,
    atlas_points_path,
    brainrender_points_path,
    abc4d_points_path,
    volume_csv_path,
    all_points_csv_path,
    summary_csv_path,
):
    source_shape = tuple(
        imio.get_size_image_from_file_paths(signal_planes).values()
    )
    source_shape = (source_shape[2], source_shape[1], source_shape[0])

    source_space = bgs.AnatomicalSpace(
        orientation,
        shape=source_shape,
        resolution=[float(i) for i in voxel_sizes],
    )

    transformed_cells = transform_points_to_atlas_space(
        cells,
        source_space,
        atlas,
        deformation_field_paths,
        downsampled_space,
        downsampled_points_path=downsampled_points_path,
        atlas_points_path=atlas_points_path,
    )

    logging.info("Summarising cell positions")
    points = summarise_points(
        cells,
        transformed_cells,
        atlas,
        volume_csv_path,
        all_points_csv_path,
        summary_csv_path,
    )
    logging.info("Exporting data")
    export_points(
        points,
        transformed_cells,
        atlas.resolution[0],
        brainrender_points_path,
        abc4d_points_path,
    )
