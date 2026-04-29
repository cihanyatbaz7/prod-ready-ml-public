"""Train a classifier to predict animal shelter outcomes."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from animal_shelter.data import load_data
from animal_shelter.features import add_features

CATEGORICAL_FEATURES = [
    "animal_type",
    "is_dog",
    "has_name",
    "sex",
    "hair_type",
]
NUMERIC_FEATURES = ["days_upon_outcome"]
TARGET = "outcome_type"


def train(data_path: str | Path, model_path: str | Path) -> Pipeline:
    """Train a model from raw data and save it to disk.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV training data.
    model_path : str or Path
        Path where the fitted model should be saved.

    Returns
    -------
    Pipeline
        The fitted scikit-learn pipeline.
    """
    data_path = Path(data_path)
    model_path = Path(model_path)
    raw_data = load_data(str(data_path))
    with_features = add_features(raw_data)
    x = with_features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = with_features[TARGET]

    pipeline = _build_pipeline()
    fitted_pipeline = _fit_model(pipeline, x, y)
    _save_model(fitted_pipeline, model_path)
    return fitted_pipeline


def _build_pipeline() -> Pipeline:
    """Build the model sklearn model pipeline."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("transformer", transformer), ("model", RandomForestClassifier())])


def _fit_model(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train the model pipeline on the given features and target."""
    pipeline.fit(x, y)
    return pipeline


def _save_model(model: Pipeline, model_path: str | Path) -> None:
    """Save the trained model to the given path."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
