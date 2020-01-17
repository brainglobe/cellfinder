import pytest

from random import randint
from argparse import ArgumentTypeError

from cellfinder.tools import misc


def test_check_positive_float():
    pos_val = randint(1, 1000) / 100
    neg_val = -randint(1, 1000) / 100

    assert pos_val == misc.check_positive_float(pos_val)

    with pytest.raises(ArgumentTypeError):
        assert misc.check_positive_float(neg_val)

    assert misc.check_positive_float(None) is None

    with pytest.raises(ArgumentTypeError):
        assert misc.check_positive_float(None, none_allowed=False)

    assert misc.check_positive_float(0) == 0


def test_check_positive_int():
    pos_val = randint(1, 1000)
    neg_val = -randint(1, 1000)

    assert pos_val == misc.check_positive_int(pos_val)

    with pytest.raises(ArgumentTypeError):
        assert misc.check_positive_int(neg_val)

    assert misc.check_positive_int(None) is None

    with pytest.raises(ArgumentTypeError):
        assert misc.check_positive_int(None, none_allowed=False)

    assert misc.check_positive_int(0) == 0
