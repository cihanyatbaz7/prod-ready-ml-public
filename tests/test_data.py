"""Unit tests for data."""

from animal_shelter import data


def test_convert_camel_case():
    """Test that camelCase strings are correctly converted to snake_case."""
    assert data.convert_camel_case("CamelCase") == "camel_case"
    assert data.convert_camel_case("CamelCASE") == "camel_case"
    assert data.convert_camel_case("camel-case") != "camel_case"
    assert data.convert_camel_case("camel_case") == "camel_case"
    assert data.convert_camel_case("camel_case ") == "camel_case "


def test_convert_camel_case_exception():
    """Test that a TypeError is raised when a non-string is passed."""
    try:
        data.convert_camel_case(None)
    except Exception as e:
        assert isinstance(e, TypeError)
