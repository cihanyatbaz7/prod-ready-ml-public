"""Utilities for loading and processing data with specific formatting requirements.

This module provides functionality for loading data from a CSV file, converting
column names, and handling missing values. Additionally, it includes utilities
to transform CamelCase strings into snake_case strings.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger("animal_shelter")


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
    logger.info(f"Loading data from {path}")

    df = (
        pd.read_csv(path, parse_dates=["DateTime"])
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=convert_camel_case)
        .fillna("Unknown")
    )

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def convert_camel_case(name: str) -> str:
    """Convert camelCaseString to snake_case_string."""
    logger.debug(f"Converting column name: {name}")
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    converted = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    logger.debug(f"Converted to: {converted}")
    return converted
