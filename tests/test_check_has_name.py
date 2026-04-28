"""Tests for verifying functionality in the animal_shelter features module."""

import pandas as pd
from animal_shelter import features
from pandas.testing import assert_series_equal


def test_check_has_name():
    """Test if the animal has a name."""
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features._check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)
