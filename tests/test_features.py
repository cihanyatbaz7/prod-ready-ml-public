"""Tests for the animal_shelter.features module."""

import pandas as pd
import pytest
from animal_shelter.features import (
    _check_has_name,
    _check_is_dog,
    _compute_days_upon_outcome,
    _get_hair_type,
    _get_neutered,
    _get_sex,
    add_features,
)
from pandas.testing import assert_series_equal


@pytest.fixture(scope="module")
def sex_upon_outcome_data() -> dict[str, pd.Series]:
    """Fixture providing mocked sex_upon_outcome data for testing."""
    return {
        "neutered_spayed": pd.Series(["Neutered Male", "Spayed Female"]),
        "intact": pd.Series(["Intact Male", "Intact Female"]),
        "unknown": pd.Series(["Unknown", "gibberish"]),
        "mixed": pd.Series(["Neutered Male", "Intact Female", "Unknown", "gibberish"]),
    }


class TestCheckIsDog:
    """Tests for _check_is_dog function."""

    def test_check_is_dog_with_dogs(self) -> None:
        """Test that dogs are correctly identified."""
        animal_type = pd.Series(["Dog", "Dog", "Cat"])
        expected = pd.Series([True, True, False])
        result = _check_is_dog(animal_type)
        assert_series_equal(result, expected)

    def test_check_is_dog_case_insensitive(self) -> None:
        """Test that check is case insensitive."""
        animal_type = pd.Series(["dog", "DOG", "Cat", "CAT"])
        expected = pd.Series([True, True, False, False])
        result = _check_is_dog(animal_type)
        assert_series_equal(result, expected)

    def test_check_is_dog_raises_on_invalid_animal(self) -> None:
        """Test that RuntimeError is raised for non-dog/cat animals."""
        animal_type = pd.Series(["Dog", "Bird"])
        with pytest.raises(RuntimeError, match="Found pets that are not dogs or cats"):
            _check_is_dog(animal_type)


class TestCheckHasName:
    """Tests for _check_has_name function."""

    def test_check_has_name_with_names(self) -> None:
        """Test that named animals are identified."""
        name = pd.Series(["Fluffy", "Max", "Unknown"])
        expected = pd.Series([True, True, False])
        result = _check_has_name(name)
        assert_series_equal(result, expected)

    def test_check_has_name_case_insensitive(self) -> None:
        """Test that check is case insensitive."""
        name = pd.Series(["Fluffy", "UNKNOWN", "unknown", "Max"])
        expected = pd.Series([True, False, False, True])
        result = _check_has_name(name)
        assert_series_equal(result, expected)


class TestGetSex:
    """Tests for _get_sex function."""

    def test_get_sex_male_female(
        self, sex_upon_outcome_data: dict[str, pd.Series]
    ) -> None:
        """Test extraction of male and female."""
        sex_upon_outcome = sex_upon_outcome_data["neutered_spayed"]
        expected = pd.Series(["male", "female"])
        result = _get_sex(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_sex_intact(self, sex_upon_outcome_data: dict[str, pd.Series]) -> None:
        """Test extraction of sex from intact animals."""
        sex_upon_outcome = sex_upon_outcome_data["intact"]
        expected = pd.Series(["male", "female"])
        result = _get_sex(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_sex_unknown(self, sex_upon_outcome_data: dict[str, pd.Series]) -> None:
        """Test that unknown sex is handled."""
        sex_upon_outcome = sex_upon_outcome_data["unknown"]
        expected = pd.Series(["unknown", "unknown"])
        result = _get_sex(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_sex_mixed(self, sex_upon_outcome_data: dict[str, pd.Series]) -> None:
        """Test mixed cases."""
        sex_upon_outcome = sex_upon_outcome_data["mixed"]
        expected = pd.Series(["male", "female", "unknown", "unknown"])
        result = _get_sex(sex_upon_outcome)
        assert_series_equal(result, expected)


class TestGetNeutered:
    """Tests for _get_neutered function."""

    def test_get_neutered_fixed(
        self, sex_upon_outcome_data: dict[str, pd.Series]
    ) -> None:
        """Test identification of neutered/spayed animals."""
        sex_upon_outcome = sex_upon_outcome_data["neutered_spayed"]
        expected = pd.Series(["fixed", "fixed"])
        result = _get_neutered(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_neutered_intact(
        self, sex_upon_outcome_data: dict[str, pd.Series]
    ) -> None:
        """Test identification of intact animals."""
        sex_upon_outcome = sex_upon_outcome_data["intact"]
        expected = pd.Series(["intact", "intact"])
        result = _get_neutered(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_neutered_unknown(
        self, sex_upon_outcome_data: dict[str, pd.Series]
    ) -> None:
        """Test that unknown status is handled."""
        sex_upon_outcome = sex_upon_outcome_data["unknown"]
        expected = pd.Series(["unknown", "unknown"])
        result = _get_neutered(sex_upon_outcome)
        assert_series_equal(result, expected)

    def test_get_neutered_mixed(
        self, sex_upon_outcome_data: dict[str, pd.Series]
    ) -> None:
        """Test mixed cases."""
        sex_upon_outcome = sex_upon_outcome_data["mixed"]
        expected = pd.Series(["fixed", "intact", "unknown", "unknown"])
        result = _get_neutered(sex_upon_outcome)
        assert_series_equal(result, expected)


class TestGetHairType:
    """Tests for _get_hair_type function."""

    def test_get_hair_type_valid_types(self) -> None:
        """Test identification of valid hair types."""
        breed = pd.Series(["Shorthair Mix", "Longhair Mix", "Medium Hair Mix"])
        expected = pd.Series(["shorthair", "longhair", "medium hair"])
        result = _get_hair_type(breed)
        assert_series_equal(result, expected)

    def test_get_hair_type_case_insensitive(self) -> None:
        """Test that hair type detection is case insensitive."""
        breed = pd.Series(["SHORTHAIR Mix", "LongHair Mix", "MEDIUM HAIR Mix"])
        expected = pd.Series(["shorthair", "longhair", "medium hair"])
        result = _get_hair_type(breed)
        assert_series_equal(result, expected)

    def test_get_hair_type_unknown(self) -> None:
        """Test that unknown hair types default to 'unknown'."""
        breed = pd.Series(["Hairless", "Curly", "Unknown"])
        expected = pd.Series(["unknown", "unknown", "unknown"])
        result = _get_hair_type(breed)
        assert_series_equal(result, expected)

    def test_get_hair_type_mixed(self) -> None:
        """Test mixed cases with valid and unknown types."""
        breed = pd.Series(["Shorthair Mix", "Curly Mix", "Longhair Mix", "Bald"])
        expected = pd.Series(["shorthair", "unknown", "longhair", "unknown"])
        result = _get_hair_type(breed)
        assert_series_equal(result, expected)


class TestComputeDaysUponOutcome:
    """Tests for _compute_days_upon_outcome function."""

    def test_compute_days_years(self) -> None:
        """Test conversion of years to days."""
        age_upon_outcome = pd.Series(["1 year", "2 years"])
        expected = pd.Series([365.0, 730.0])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert_series_equal(result, expected)

    def test_compute_days_months(self) -> None:
        """Test conversion of months to days."""
        age_upon_outcome = pd.Series(["1 month", "2 months"])
        expected = pd.Series([30.0, 60.0])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert_series_equal(result, expected)

    def test_compute_days_weeks(self) -> None:
        """Test conversion of weeks to days."""
        age_upon_outcome = pd.Series(["1 week", "2 weeks"])
        expected = pd.Series([7.0, 14.0])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert_series_equal(result, expected)

    def test_compute_days_days(self) -> None:
        """Test that days remain unchanged."""
        age_upon_outcome = pd.Series(["1 day", "5 days"])
        expected = pd.Series([1.0, 5.0])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert_series_equal(result, expected)

    def test_compute_days_mixed(self) -> None:
        """Test mixed time units."""
        age_upon_outcome = pd.Series(["1 year", "2 months", "3 weeks", "5 days"])
        expected = pd.Series([365.0, 60.0, 21.0, 5.0])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert_series_equal(result, expected)

    def test_compute_days_unknown(self) -> None:
        """Test that unknown ages are converted to NaN."""
        age_upon_outcome = pd.Series(["Unknown", "1 day"])
        result = _compute_days_upon_outcome(age_upon_outcome)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 1.0


class TestAddFeatures:
    """Tests for add_features function."""

    @pytest.fixture(scope="class")
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "animal_type": ["Dog", "Cat", "Dog"],
                "name": ["Fluffy", "Unknown", "Max"],
                "sex_upon_outcome": ["Neutered Male", "Spayed Female", "Intact Male"],
                "breed": ["Shorthair Mix", "Longhair Mix", "Unknown"],
                "age_upon_outcome": ["1 year", "2 months", "5 days"],
            }
        )

    def test_add_features_creates_all_columns(self, sample_data: pd.DataFrame) -> None:
        """Test that all feature columns are created."""
        result = add_features(sample_data)
        expected_columns = [
            "is_dog",
            "has_name",
            "sex",
            "neutered",
            "hair_type",
            "days_upon_outcome",
        ]
        for col in expected_columns:
            assert col in result.columns

    def test_add_features_no_side_effects(self, sample_data: pd.DataFrame) -> None:
        """Test that add_features does not modify input DataFrame."""
        original_data = sample_data.copy()
        add_features(sample_data)
        pd.testing.assert_frame_equal(sample_data, original_data)

    def test_add_features_correct_values(self, sample_data: pd.DataFrame) -> None:
        """Test that features have correct values."""
        result = add_features(sample_data)

        # Check is_dog
        assert_series_equal(
            result["is_dog"], pd.Series([True, False, True], name="is_dog")
        )

        # Check has_name
        assert_series_equal(
            result["has_name"], pd.Series([True, False, True], name="has_name")
        )

        # Check sex
        assert_series_equal(
            result["sex"],
            pd.Series(["male", "female", "male"], name="sex"),
        )

        # Check neutered
        assert_series_equal(
            result["neutered"],
            pd.Series(["fixed", "fixed", "intact"], name="neutered"),
        )

        # Check days_upon_outcome
        assert_series_equal(
            result["days_upon_outcome"],
            pd.Series([365.0, 60.0, 5.0], name="days_upon_outcome"),
        )

    def test_add_features_preserves_original_columns(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test that original columns are preserved."""
        result = add_features(sample_data)
        for col in sample_data.columns:
            assert col in result.columns
