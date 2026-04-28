"""Test the convert_camel_case function."""

import pytest
from animal_shelter import data


class TestConvertCamelCase:
    """Tests for converting camel case to snake case."""

    def test_convert_camel_case(self) -> None:
        """Test if the function converts camel case to snake case."""
        assert data.convert_camel_case("CamelCase") == "camel_case"
        assert data.convert_camel_case("CamelCASE") == "camel_case"
        assert data.convert_camel_case("camel-case") != "camel_case"
        assert data.convert_camel_case("camel Case") != "camel_case"
        assert data.convert_camel_case("camel_case") == "camel_case"

    def test_convert_camel_case_non_string(self) -> None:
        """Test if the function raises TypeError for non-string inputs."""
        with pytest.raises(TypeError):
            data.convert_camel_case(123)
