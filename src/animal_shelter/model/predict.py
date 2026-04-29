"""Generate predictions from a trained animal shelter model."""

from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def predict(data_path: Path, model_path: Path) -> Any:
    """Generate predictions from new data using a saved model.

    Parameters
    ----------
    data_path : Path
        Path to the CSV data to predict on.
    model_path : Path
        Path to the saved fitted model.

    Returns
    -------
    np.ndarray
        Array of predicted outcomes.
    """
    model = _load_model(model_path)
    raw_data = load_data(str(data_path))
    with_features = add_features(raw_data)
    x = with_features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    return model.predict(x)


def _load_model(model_path: Path) -> Pipeline:
    """Load a fitted model from disk."""
    return joblib.load(model_path)
