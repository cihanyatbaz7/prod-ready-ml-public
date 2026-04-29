"""Tests for the model training and prediction modules."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from animal_shelter.model.predict import predict
from animal_shelter.model.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _build_pipeline,
    train,
)

# -------------- Unit tests for _build_pipeline --------------


def test_build_pipeline_returns_pipeline():
    """_build_pipeline returns a sklearn Pipeline."""
    pipeline = _build_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_has_correct_steps():
    """_build_pipeline contains 'transformer' and 'model' steps."""
    pipeline = _build_pipeline()
    assert "transformer" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_pipeline_uses_random_forest():
    """_build_pipeline uses a RandomForestClassifier."""
    pipeline = _build_pipeline()
    model = pipeline.named_steps["model"]
    assert isinstance(model, RandomForestClassifier)


def test_build_pipeline_uses_correct_feature_columns():
    """_build_pipeline's ColumnTransformer uses the correct feature columns."""
    pipeline = _build_pipeline()
    transformer = pipeline.named_steps["transformer"]
    assert isinstance(transformer, ColumnTransformer)

    # Check features
    assert set(transformer.transformers[0][2]) == set(NUMERIC_FEATURES)
    assert set(transformer.transformers[1][2]) == set(CATEGORICAL_FEATURES)


# -------------- Smoke tests for train and predict --------------


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a tiny but realistic CSV to a temporary file and return its path."""
    df = pd.DataFrame(
        {
            "AnimalID": ["A1", "A2", "A3", "A4"],
            "Name": ["Buddy", "unknown", "Whiskers", "Max"],
            "DateTime": [
                "2014-02-15 10:00:00",
                "2014-02-15 11:00:00",
                "2014-02-15 12:00:00",
                "2014-02-15 13:00:00",
            ],
            "OutcomeType": ["Adoption", "Transfer", "Adoption", "Transfer"],
            "AnimalType": ["Dog", "Dog", "Cat", "Dog"],
            "SexuponOutcome": [
                "Neutered Male",
                "Spayed Female",
                "Intact Male",
                "Neutered Male",
            ],
            "AgeuponOutcome": ["2 years", "5 months", "3 weeks", "1 year"],
            "Breed": [
                "Labrador Retriever Mix",
                "Poodle Mix",
                "Domestic Shorthair Mix",
                "Beagle Shorthair",
            ],
            "Color": ["Brown", "White", "Black", "Tan"],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_train_creates_model_file(sample_csv: Path, tmp_path: Path):
    """Train writes a model file to the given path."""
    model_path = tmp_path / "output" / "model.pkl"
    train(sample_csv, model_path)
    assert model_path.exists()


def test_train_returns_fitted_pipeline(sample_csv: Path, tmp_path: Path):
    """Train writes a model file to the given path."""
    model_path = tmp_path / "output" / "model.pkl"
    pipeline = train(sample_csv, model_path)
    assert isinstance(pipeline, Pipeline)


def test_predict_returns_one_prediction_per_row(sample_csv: Path, tmp_path: Path):
    """Predict returns one prediction per row in the input data."""
    model_path = tmp_path / "output" / "model.pkl"
    train(sample_csv, model_path)
    predictions = predict(sample_csv, model_path)

    expected_count = len(pd.read_csv(sample_csv))
    assert len(predictions) == expected_count
