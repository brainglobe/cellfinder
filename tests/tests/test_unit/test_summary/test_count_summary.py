import random
import pytest
import argparse

from cellfinder.summarise import count_summary


def test_point_coordinates():
    x = random.randint(0, 1000)
    y = random.randint(0, 1000)
    string = str(x) + "," + str(y)
    assert (x, y) == count_summary.point_coordinates(string)

    string_space = " " + str(x) + " , " + str(y)
    assert (x, y) == count_summary.point_coordinates(string_space)

    with pytest.raises(argparse.ArgumentTypeError):
        count_summary.point_coordinates("random string with numbers 10 12")
