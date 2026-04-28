"""Module for making predictions with the trained animal shelter model."""

from pathlib import Path
from typing import cast

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from animal_shelter.data import load_data
from animal_shelter.features import add_features


def predict(data_path: str | Path, model_path: str | Path) -> np.ndarray:
    """Make predictions on new data using a trained model.

    Parameters
    ----------
    data_path : str | Path
        Path to the data CSV file for prediction
    model_path : str | Path
        Path to the trained model file

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels

    """
    # Convert to Path objects for consistency
    data_path = Path(data_path)
    model_path = Path(model_path)

    # Load and process data
    raw_data = load_data(str(data_path))
    with_features = add_features(raw_data)

    # Extract features (same as in training)
    cat_features = [
        "animal_type",
        "is_dog",
        "has_name",
        "sex",
        "hair_type",
    ]
    num_features = ["days_upon_outcome"]

    x = with_features[cat_features + num_features]

    # Load model and make predictions
    model = _load_model(model_path)
    predictions = cast(np.ndarray, model.predict(x))

    return predictions


def _load_model(model_path: Path) -> Pipeline:
    """Load a trained model from disk.

    Parameters
    ----------
    model_path : Path
        Path to the trained model file

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Loaded model pipeline

    """
    model: Pipeline = joblib.load(model_path)
    return model
