import pandas as pd
import pytest

from pandas.testing import assert_series_equal
from animal_shelter import features

def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features.check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)

def test_get_hair_type():
    s = pd.Series(["unknown", "shorthair", "medium hair", "longhair"])
    result = features.get_hair_type(s)
    expected = pd.Series(["unknown", "shorthair", "medium hair", "longhair"])
    assert_series_equal(result, expected)

def test_check_is_dog():
    s = pd.Series(["dog", "Dog", "cat"])
    result = features.check_is_dog(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


@pytest.fixture(scope="class")
def list_of_sex_outcomes():
    return pd.Series(["Neutered Male", "Spayed Female", "Intact Female", "Intact Male"])

class TestListFunctions:
    def test_get_sex(self, list_of_sex_outcomes):
        result = features.get_sex(list_of_sex_outcomes)
        expected = pd.Series(["male", "female", "female", "male"])
        assert_series_equal(result, expected)

    def test_get_neutered(self, list_of_sex_outcomes):
        result = features.get_neutered(list_of_sex_outcomes)
        expected = pd.Series(["fixed", "fixed", "intact", "intact"])
        assert_series_equal(result, expected)
