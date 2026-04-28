"""Unit tests for features."""

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from animal_shelter import features


@pytest.fixture()
def sex_upon_outcome():
    """Provide a sample Series of sex-upon-outcome strings for testing."""
    return pd.Series(
        ["neutered Male", "spayed Female", "intact Male", "unknown Female"],
    )


def test_check_has_name():
    """Test that known names return True and 'unknown' returns False."""
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features._check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


def test_check_is_dog():
    """Test that dog entries (case-insensitive) return True and others False."""
    s = pd.Series(["Dog", "Cat", "dog", "cat"])
    result = features._check_is_dog(s)
    expected = pd.Series([True, False, True, False])
    assert_series_equal(result, expected)


def test_get_sex(sex_upon_outcome):
    """Test that the sex is correctly extracted from the sex-upon-outcome field."""
    result = features._get_sex(sex_upon_outcome)
    expected = pd.Series(["male", "female", "male", "female"])
    assert_series_equal(result, expected)


def test_get_neutered(sex_upon_outcome):
    """Test that neutered status is correctly classified as fixed, intact, or unknown."""
    result = features._get_neutered(sex_upon_outcome)
    expected = pd.Series(["fixed", "fixed", "intact", "unknown"])
    assert_series_equal(result, expected)


def test_get_hair_type():
    """Test that recognised hair types are returned and unrecognised ones become 'unknown'."""
    s = pd.Series(["shorthair", "medium hair", "medium", "longhair"])
    result = features._get_hair_type(s)
    expected = pd.Series(["shorthair", "medium hair", "unknown", "longhair"])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome():
    """Test that time-upon-outcome strings are correctly converted to days."""
    s = pd.Series(["1 year", "6 months", "3 weeks", "2 days"])
    result = features._compute_days_upon_outcome(s)
    expected = pd.Series([365.0, 180.0, 21.0, 2.0])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome_exception():
    """Test that an IndexError is raised for malformed time strings."""
    try:
        s = pd.Series(["week", "1 year", "6 months"])
        features._compute_days_upon_outcome(s)
    except Exception as e:
        assert isinstance(e, IndexError)
