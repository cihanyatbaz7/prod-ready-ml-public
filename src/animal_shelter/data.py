"""Data loading utilities for the animal shelter dataset."""

import re

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    return (
        pd.read_csv(path, parse_dates=["DateTime"])
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=_convert_camel_case)
        .fillna("Unknown")
    )


def _convert_camel_case(name: str) -> str:
    """Convert camelCaseString to snake_case_string."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
