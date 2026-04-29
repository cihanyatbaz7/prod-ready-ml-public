"""Tests for the features module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from animal_shelter import features


def test_check_is_dog():
    """Test that dogs are flagged as True and cats as False."""
    s = pd.Series(["Dog", "Cat", "dog", "cat"])
    result = features._check_is_dog(s)
    expected = pd.Series([True, False, True, False])
    assert_series_equal(result, expected)


def test_check_is_dog_raises_on_non_cat_dog():
    """Test that passing a non-cat/dog value raises RuntimeError."""
    s = pd.Series(["Dog", "Cat", "Rabbit"])
    with pytest.raises(RuntimeError) as exception:
        features._check_is_dog(s)
    assert "not dogs or cats" in str(exception.value).lower()


def test_check_has_name():
    """Test that animals named 'unknown' are flagged as having no name."""
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features._check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


@pytest.fixture()
def sex_upon_outcome_series():
    """Fixture for testing the _get_sex function."""
    return pd.Series(["Neutered Male", "Spayed Female", "Intact Male", "Unknown"])


def test_get_sex(sex_upon_outcome_series):
    """Test that sex is correctly extracted from 'sex_upon_outcome' values."""
    result = features._get_sex(sex_upon_outcome_series)
    expected = pd.Series(["male", "female", "male", "unknown"])
    assert_series_equal(result, expected)


def test_get_neutered(sex_upon_outcome_series):
    """Test that neuter status is correctly classified as fixed/intact/unknown."""
    result = features._get_neutered(sex_upon_outcome_series)
    expected = pd.Series(["fixed", "fixed", "intact", "unknown"])
    assert_series_equal(result, expected)


def test_get_hair_type():
    """Test that hair type is correctly extracted from breed strings."""
    s = pd.Series(
        [
            "Domestic Shorthair Mix",
            "Domestic Medium Hair",
            "Domestic Longhair",
            "Labrador Retriever",
        ]
    )
    result = features._get_hair_type(s)
    expected = pd.Series(["shorthair", "medium hair", "longhair", "unknown"])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome():
    """Test that age strings are correctly converted to age in days."""
    s = pd.Series(["1 year", "2 weeks", "3 months", "5 days", "Unknown"])
    result = features._compute_days_upon_outcome(s)
    expected = pd.Series([365.0, 14.0, 90.0, 5.0, np.nan])
    assert_series_equal(result, expected)
