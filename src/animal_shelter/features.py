"""Feature engineering for the animal shelter dataset."""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add some features to our data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with data (see load_data)

    Returns
    -------
    with_features : pandas.DataFrame
        DataFrame with some column features added
    """
    return df.assign(
        is_dog=_check_is_dog(df["animal_type"]),
        has_name=_check_has_name(df["name"]),
        sex=_get_sex(df["sex_upon_outcome"]),
        neutered=_get_neutered(df["sex_upon_outcome"]),
        hair_type=_get_hair_type(df["breed"]),
        days_upon_outcome=_compute_days_upon_outcome(df["age_upon_outcome"]),
    )


def _check_is_dog(animal_type: pd.Series) -> pd.Series:
    """Check if the animal is a dog, otherwise return False."""
    is_cat_dog = animal_type.str.lower().isin(["dog", "cat"])
    if not is_cat_dog.all():
        print(
            "Found something else but dogs and cats:\n%s",
            animal_type[~is_cat_dog],
        )
        raise RuntimeError("Found pets that are not dogs or cats.")
    return animal_type.str.lower() == "dog"


def _check_has_name(name: pd.Series) -> pd.Series:
    """Check if the animal is not called 'unknown'."""
    return name.str.lower() != "unknown"


def _get_sex(sex_upon_outcome: pd.Series) -> pd.Series:
    """Determine if the sex was 'male', 'female' or 'unknown'."""
    sex = pd.Series("unknown", index=sex_upon_outcome.index)
    sex.loc[sex_upon_outcome.str.endswith("Female")] = "female"
    sex.loc[sex_upon_outcome.str.endswith("Male")] = "male"
    return sex


def _get_neutered(sex_upon_outcome: pd.Series) -> pd.Series:
    """Determine whether the animal is fixed, intact, or unknown."""
    neutered = sex_upon_outcome.str.lower().copy()
    neutered.loc[neutered.str.contains("neutered")] = "fixed"
    neutered.loc[neutered.str.contains("spayed")] = "fixed"
    neutered.loc[neutered.str.contains("intact")] = "intact"
    neutered.loc[~neutered.isin(["fixed", "intact"])] = "unknown"
    return neutered


def _get_hair_type(breed: pd.Series) -> pd.Series:
    """Determine the hair type of the animal based on its breed."""
    hair_type = breed.str.lower().copy()
    valid_hair_types = ["shorthair", "medium hair", "longhair"]
    for hair in valid_hair_types:
        is_hair_type = hair_type.str.contains(hair)
        hair_type[is_hair_type] = hair
    hair_type[~hair_type.isin(valid_hair_types)] = "unknown"
    return hair_type


def _compute_days_upon_outcome(age_upon_outcome: pd.Series) -> pd.Series:
    """Convert age strings (e.g. '2 years', '3 weeks') into age in days."""
    split_age = age_upon_outcome.str.split()
    time = split_age.apply(lambda x: x[0] if x[0] != "Unknown" else np.nan)
    period = split_age.apply(lambda x: x[1] if x[0] != "Unknown" else None)
    period_mapping = {
        "year": 365,
        "years": 365,
        "month": 30,
        "months": 30,
        "week": 7,
        "weeks": 7,
        "day": 1,
        "days": 1,
    }
    return time.astype(float) * period.map(period_mapping)
