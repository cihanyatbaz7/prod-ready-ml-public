import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from animal_shelter import features

#maak fixture aan voor de tests om te kunnen hergebruiken, zodat je niet steeds dezelfde code hoeft te schrijven in elke test.
@pytest.fixture()
def animal_type():
    return pd.Series(["Dog", "Cat", "Dog"])

#Dit is mocked input data: zelfgemaakte testdata waarmee je de functies test, zonder dat je daarvoor een echt databestand hoeft in te laden.
@pytest.fixture()
def sex_upon_outcome():
    return pd.Series([
        "Neutered Male",
        "Spayed Female",
        "Intact Male",
        "Unknown"
    ])

def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features.test_check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)

def test_check_is_dog(animal_type):
    result = features._check_is_dog(animal_type)

    expected = pd.Series([True, False, True])

    pd.testing.assert_series_equal(result, expected)

def test_check_is_dog(animal_type):
    # animal_type = pd.Series(["Dog", "Cat", "Dog"]) #fixture is used instead of this line

    result = features._check_is_dog(animal_type)

    expected = pd.Series([True, False, True])

    pd.testing.assert_series_equal(result, expected)

def test_check_is_dog_length(animal_type):
    result = features._check_is_dog(animal_type)

    assert len(result) == 3

def test_check_is_dog_invalid_animal():
    animal_type = pd.Series(["Dog", "Bird"])

    with pytest.raises(RuntimeError):
        features._check_is_dog(animal_type)


def test_check_has_name():
    name = pd.Series(["Ivo", "Henk", "unknown", "UNKNOWN"])

    result = features._check_has_name(name)

    assert result.tolist() == [True, True, False, False]


# def test_get_sex():
#     sex_upon_outcome = pd.Series([
#         "Neutered Male",
#         "Spayed Female",
#         "Unknown"
#     ])

#     result = features._get_sex(sex_upon_outcome)

#     assert result.tolist() == ["male", "female", "unknown"]

#nu met fixture kunnen we het zo opzetten:
def test_get_sex(sex_upon_outcome):
    result = features._get_sex(sex_upon_outcome)

    expected = pd.Series([
        "male",
        "female",
        "male",
        "unknown"
    ])

    pd.testing.assert_series_equal(result, expected)

# def test_get_neutered():
#     sex_upon_outcome = pd.Series([
#         "Neutered Male",
#         "Spayed Female",
#         "Intact Male",
#         "Unknown"
#     ])

#     result = features._get_neutered(sex_upon_outcome)

#     assert result.tolist() == ["fixed", "fixed", "intact", "unknown"]

# deze kan ook middels de fixutre (ze hebben nu beide een gedeelde testset)
def test_get_neutered(sex_upon_outcome):
    result = features._get_neutered(sex_upon_outcome)

    expected = pd.Series([
        "fixed",
        "fixed",
        "intact",
        "unknown"
    ])

    pd.testing.assert_series_equal(result, expected)

def test_get_hair_type():
    breed = pd.Series([
        "Domestic Shorthair Mix",
        "Domestic Medium Hair",
        "Domestic Longhair Mix",
        "Siamese"
    ])

    result = features._get_hair_type(breed)

    assert result.tolist() == [
        "shorthair",
        "medium hair",
        "longhair",
        "unknown"
    ]


def test_compute_days_upon_outcome():
    age_upon_outcome = pd.Series([
        "2 years",
        "3 months",
        "4 weeks",
        "5 days",
        "Unknown"
    ])

    result = features._compute_days_upon_outcome(age_upon_outcome)

    assert result.iloc[:4].tolist() == [730, 90, 28, 5]
    assert pd.isna(result.iloc[4])

# test exceptions
def test_check_is_dog_invalid_animal():
    animal_type = pd.Series(["Dog", "Bird"])

    with pytest.raises(RuntimeError) as exception:
        features._check_is_dog(animal_type)

    assert "Found pets that are not dogs or cats." in str(exception.value)


def test_compute_days_upon_outcome_invalid_input():
    age_upon_outcome = pd.Series(["2"])

    with pytest.raises(IndexError):
        features._compute_days_upon_outcome(age_upon_outcome)
