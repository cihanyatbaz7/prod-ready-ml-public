"""Tests for the data module."""

import pytest

from animal_shelter import data


def test_convert_camel_case():
    """Test that camelCase strings are converted to snake_case."""
    assert data._convert_camel_case("CamelCase") == "camel_case"
    assert data._convert_camel_case("CamelCASE") == "camel_case"
    assert data._convert_camel_case("camel-case") != "camel_case"


def test_convert_camel_case_raises_on_non_string():
    """Test that passing a non-string value raises TypeError."""
    with pytest.raises(TypeError) as exception:
        data._convert_camel_case(123)
    assert "string" in str(exception.value).lower()
