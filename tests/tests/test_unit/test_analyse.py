import filecmp
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
        structure="Anterodorsal nucleus",
        structure_id=57,
        hemisphere="left",
    )
    return [p1, p2]


@pytest.fixture
def structures_with_points():
    return [
        "Paraventricular hypothalamic nucleus, descending division",
        "Anterodorsal nucleus",
    ]


def test_get_region_totals(points, structures_with_points):
    """Regression test for get_region_totals"""
    volumes_path = Path("./tests/data/analyse/volumes.csv")
    output_path = Path("./tests/data/analyse/region_totals.csv")
    output_path.unlink(missing_ok=True)
    get_region_totals(
        points, structures_with_points, volumes_path, output_path
    )
    assert output_path.exists()
    expected_output = Path(
        "./tests/data/analyse/region_totals_regression_pandas1_5_3.csv"
    )
    assert filecmp.cmp(output_path, expected_output)
