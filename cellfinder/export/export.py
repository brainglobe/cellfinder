from cellfinder.export.brainrender import export_points as brainrender_export
from cellfinder.export.abc4d import export_points as abc4d_export


def export_points(
    point_info, points, resolution, brainrender_points_path, abc4d_points_path
):
    brainrender_export(points, resolution, brainrender_points_path)
    abc4d_export(point_info, resolution, abc4d_points_path)
