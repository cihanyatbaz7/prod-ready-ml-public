"""Animal shelter package: data loading and feature engineering for outcome prediction."""

from animal_shelter.data import load_data
from animal_shelter.features import add_features

__all__ = ["load_data", "add_features"]


def main() -> None:
    """Print a greeting from the animal-shelter package."""
    print("Hello from animal-shelter!")
