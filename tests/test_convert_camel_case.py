"""Test the convert_camel_case function."""

from animal_shelter import data


def test_convert_camel_case():
    """Test if the function converts camel case to snake case."""
    assert data.convert_camel_case("CamelCase") == "camel_case"
    assert data.convert_camel_case("CamelCASE") == "camel_case"
    assert data.convert_camel_case("camel-case") != "camel_case"
    assert data.convert_camel_case("camel Case") != "camel_case"
    assert data.convert_camel_case("camel_case") == "camel_case"
