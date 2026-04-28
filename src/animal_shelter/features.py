"""Module for processing and augmenting animal shelter data.

This module provides functionality to add features to animal shelter data,
such as determining if an animal is a dog, calculating age in days, and
identifying attributes like hair type, sex, or neutered status. It ensures
input data integrity and calculates relevant features that can be used for
further analysis.

"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("animal_shelter")


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
    logger.info("Adding features to data")

    original_columns = set(df.columns)

    # Use .copy() to avoid side effects on input DataFrame
    result = df.copy()
    result["is_dog"] = _check_is_dog(result["animal_type"])
    result["has_name"] = _check_has_name(result["name"])
    result["sex"] = _get_sex(result["sex_upon_outcome"])
    result["neutered"] = _get_neutered(result["sex_upon_outcome"])
    result["hair_type"] = _get_hair_type(result["breed"])
    result["days_upon_outcome"] = _compute_days_upon_outcome(result["age_upon_outcome"])

    new_columns = set(result.columns) - original_columns
    logger.info(f"Successfully added {len(new_columns)} new features")
    logger.debug(f"New features: {sorted(new_columns)}")

    return result


def _check_is_dog(animal_type: pd.Series) -> pd.Series:
    """Check if the animal is a dog, otherwise return False.

    Parameters
    ----------
    animal_type : pandas.Series
        Type of animal

    Returns
    -------
    result : pandas.Series
        Dog or not

    """
    logger.debug("Checking if animals are dogs")

    # Check if it's either a cat or a dog.
    is_cat_dog = animal_type.str.lower().isin(["dog", "cat"])
    if not is_cat_dog.all():
        invalid_animals = animal_type[~is_cat_dog].unique()
        logger.warning(f"Found non-dog/cat animals: {invalid_animals}")
        raise RuntimeError("Found pets that are not dogs or cats.")

    is_dog = animal_type.str.lower() == "dog"
    logger.debug(f"Found {is_dog.sum()} dogs out of {len(is_dog)} animals")

    return is_dog


def _check_has_name(name: pd.Series) -> pd.Series:
    """Check if the animal is not called 'unknown'.

    Parameters
    ----------
    name : pandas.Series
        Animal name

    Returns
    -------
    result : pandas.Series
        Unknown or not.

    """
    logger.debug("Checking which animals have names")

    has_name = name.str.lower() != "unknown"
    logger.debug(f"Found {has_name.sum()} named animals out of {len(has_name)} total")

    return has_name


def _get_sex(sex_upon_outcome: pd.Series) -> pd.Series:
    """Determine if the sex was 'Male', 'Female' or unknown.

    Parameters
    ----------
    sex_upon_outcome : pandas.Series
        Sex and fixed state when coming in

    Returns
    -------
    sex : pandas.Series
        Sex when coming in

    """
    logger.debug("Extracting sex from sex_upon_outcome")

    sex = pd.Series("unknown", index=sex_upon_outcome.index)
    sex.loc[sex_upon_outcome.str.endswith("Female")] = "female"
    sex.loc[sex_upon_outcome.str.endswith("Male")] = "male"

    logger.debug(
        f"Sex distribution - Male: {(sex == 'male').sum()}, "
        f"Female: {(sex == 'female').sum()}, Unknown: {(sex == 'unknown').sum()}"
    )

    return sex


def _get_neutered(sex_upon_outcome: pd.Series) -> pd.Series:
    """Determine if an animal was intact or not.

    Parameters
    ----------
    sex_upon_outcome : pandas.Series
        Sex and fixed state when coming in

    Returns
    -------
    neutered : pandas.Series
        Intact, fixed or unknown

    """
    logger.debug("Extracting neutered status from sex_upon_outcome")

    neutered = sex_upon_outcome.str.lower().copy()
    neutered.loc[neutered.str.contains("neutered")] = "fixed"
    neutered.loc[neutered.str.contains("spayed")] = "fixed"
    neutered.loc[neutered.str.contains("intact")] = "intact"
    neutered.loc[~neutered.isin(["fixed", "intact"])] = "unknown"

    logger.debug(
        f"Neutered status - Fixed: {(neutered == 'fixed').sum()}, "
        f"Intact: {(neutered == 'intact').sum()}, Unknown: {(neutered == 'unknown').sum()}"
    )

    return neutered


def _get_hair_type(breed: pd.Series) -> pd.Series:
    """Get hair type of a breed.

    Parameters
    ----------
    breed : pandas.Series
        Breed of animal

    Returns
    -------
    hair_type : pandas.Series
        Hair type

    """
    logger.debug("Extracting hair type from breed")

    hair_type = breed.str.lower().copy()
    valid_hair_types = ["shorthair", "medium hair", "longhair"]

    for hair in valid_hair_types:
        is_hair_type = hair_type.str.contains(hair)
        hair_type.loc[is_hair_type] = hair

    hair_type.loc[~hair_type.isin(valid_hair_types)] = "unknown"

    logger.debug(
        f"Hair type distribution - Shorthair: {(hair_type == 'shorthair').sum()}, "
        f"Medium hair: {(hair_type == 'medium hair').sum()}, "
        f"Longhair: {(hair_type == 'longhair').sum()}, "
        f"Unknown: {(hair_type == 'unknown').sum()}"
    )

    return hair_type


def _compute_days_upon_outcome(age_upon_outcome: pd.Series) -> pd.Series:
    """Compute age in days upon outcome.

    Parameters
    ----------
    age_upon_outcome : pandas.Series
        Age as string

    Returns
    -------
    days_upon_outcome : pandas.Series
        Age in days

    """
    logger.debug("Computing days upon outcome from age strings")

    split_age = age_upon_outcome.str.split()
    time = split_age.apply(lambda x: x[0] if x[0] != "Unknown" else np.nan)
    period = split_age.apply(lambda x: x[1] if x[0] != "Unknown" else None)
    period_mapping = {
        "year": 365,
        "years": 365,
        "weeks": 7,
        "week": 7,
        "month": 30,
        "months": 30,
        "days": 1,
        "day": 1,
    }
    days_upon_outcome = time.astype(float) * period.map(period_mapping)

    logger.debug(
        f"Computed days - Mean: {days_upon_outcome.mean():.2f}, "
        f"Min: {days_upon_outcome.min():.2f}, Max: {days_upon_outcome.max():.2f}, "
        f"Missing values: {days_upon_outcome.isna().sum()}"
    )

    return days_upon_outcome
