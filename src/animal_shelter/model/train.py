"""Module for training the animal shelter classification model."""

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


def train(data_path: str | Path, model_path: str | Path) -> None:
    """Train a classification model and save it to disk.

    Parameters
    ----------
    data_path : str | Path
        Path to the training data CSV file
    model_path : str | Path
        Path where the trained model should be saved

    """
    # Convert to Path objects for consistency
    data_path = Path(data_path)
    model_path = Path(model_path)

    # Load and process data
    raw_data = load_data(str(data_path))
    with_features = add_features(raw_data)

    # Extract features and target
    cat_features = [
        "animal_type",
        "is_dog",
        "has_name",
        "sex",
        "hair_type",
    ]
    num_features = ["days_upon_outcome"]

    x = with_features[cat_features + num_features]
    y = with_features["outcome_type"]

    # Build and fit model
    clf_model = _build_pipeline(cat_features, num_features)
    _fit_model(clf_model, x, y)

    # Save model
    _save_model(clf_model, model_path)


def _build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    """Build the sklearn preprocessing and model pipeline.

    Parameters
    ----------
    cat_features : list[str]
        List of categorical feature column names
    num_features : list[str]
        List of numerical feature column names

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Complete pipeline with preprocessing and classifier

    """
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )

    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    transformer = ColumnTransformer(
        (
            ("numeric", num_transformer, num_features),
            ("categorical", cat_transformer, cat_features),
        )
    )

    clf_model = Pipeline(
        [("transformer", transformer), ("model", RandomForestClassifier())]
    )

    return clf_model


def _fit_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the model pipeline on training data.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The pipeline to fit
    x : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Fitted model pipeline

    """
    model.fit(x, y)
    return model


def _save_model(model: Pipeline, model_path: Path) -> None:
    """Save the trained model to disk.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    model_path : Path
        Path where the model should be saved

    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
