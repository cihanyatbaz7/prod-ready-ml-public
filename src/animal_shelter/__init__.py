"""Animal shelter package: data loading and feature engineering for outcome prediction."""

from animal_shelter.data import convert_camel_case, load_data
from animal_shelter.features import add_features

__all__ = ["load_data", "convert_camel_case", "add_features"]


def main() -> None:
    print("Hello from animal-shelter!")
