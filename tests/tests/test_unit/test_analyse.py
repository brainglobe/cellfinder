from pathlib import Path

import pytest

from cellfinder.analyse.analyse import Point, get_region_totals


@pytest.fixture
def points():
    p1 = Point(
        raw_coordinate=[1.0, 1.0, 1.0],
        atlas_coordinate=[1.0, 1.0, 1.0],
        structure="Paraventricular hypothalamic nucleus, descending division",
        structure_id=56,
        hemisphere="right",
    )
    p2 = Point(
        raw_coordinate=[2.0, 2.0, 2.0],
        atlas_coordinate=[2.0, 2.0, 2.0],
        structure="Paraventricular hypothalamic nucleus, descending division",
        structure_id=56,
        hemisphere="right",
    )
    return [p1, p2]


@pytest.fixture
def structures_with_points():
    return ["Paraventricular hypothalamic nucleus, descending division"]


def test_get_region_totals(points, structures_with_points):
    """Smoke test for get_region_totals"""
    volumes_path = Path("./tests/data/analyse/volumes.csv")
    output_path = Path("./tests/data/analyse/region_totals.csv")
    output_path.unlink(missing_ok=True)
    get_region_totals(
        points, structures_with_points, volumes_path, output_path
    )
    assert output_path.exists()
