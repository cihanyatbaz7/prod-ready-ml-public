import pandas as pd

from animal_shelter import data
import pytest


def test_convert_camel_case():
    assert data.convert_camel_case("CamelCase") == "camel_case"
    assert data.convert_camel_case("CamelCASE") == "camel_case"
    assert data.convert_camel_case("camel_case") == "camel_case"
    assert data.convert_camel_case("AnimalType") == "animal_type"
    assert data.convert_camel_case("DateTime") == "date_time"


def test_load_data(tmp_path):
    csv_file = tmp_path / "test_data.csv"

    csv_file.write_text(
        "DateTime,AnimalType,Name,SexuponOutcome\n"
        "2020-01-01,Dog,Ivo,Neutered Male\n"
        "2020-01-02,Cat,,Spayed Female\n"
    )

    result = data.load_data(str(csv_file))

    assert "animal_type" in result.columns
    assert "sex_upon_outcome" in result.columns
    assert result["name"].tolist() == ["Ivo", "Unknown"]
    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result["date_time"])

# testing exceptions
def test_convert_camel_case_invalid_input():
    with pytest.raises(TypeError) as exception:
        data.convert_camel_case(None) #je voert bewust ongeldige input in en controleert of de functie de verwachte fout veroorzaakt.

    assert "expected string or bytes-like object" in str(exception.value)
