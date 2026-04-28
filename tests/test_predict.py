"""Tests for the animal_shelter.model.predict module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from animal_shelter.model.predict import _load_model
from animal_shelter.model.train import _build_pipeline, _fit_model, _save_model
from sklearn.pipeline import Pipeline


@pytest.fixture
def cat_features() -> list[str]:
    """Fixture providing categorical feature names."""
    return ["animal_type", "is_dog", "sex"]


@pytest.fixture
def num_features() -> list[str]:
    """Fixture providing numerical feature names."""
    return ["days_upon_outcome"]


@pytest.fixture
def sample_x() -> pd.DataFrame:
    """Fixture providing sample feature data."""
    return pd.DataFrame(
        {
            "animal_type": ["Dog", "Cat"] * 10,
            "is_dog": [True, False] * 10,
            "sex": ["male", "female"] * 10,
            "days_upon_outcome": [10.0, 20.0] * 10,
        }
    )


@pytest.fixture
def sample_y() -> pd.Series:
    """Fixture providing sample target data."""
    return pd.Series(["Adoption", "Return_to_owner"] * 10)


@pytest.fixture
def trained_model_path(
    cat_features: list[str],
    num_features: list[str],
    sample_x: pd.DataFrame,
    sample_y: pd.Series,
    tmp_path: Path,
) -> Path:
    """Fixture providing a trained model saved to disk."""
    pipeline = _build_pipeline(cat_features, num_features)
    fitted = _fit_model(pipeline, sample_x, sample_y)
    model_path = tmp_path / "model.pkl"
    _save_model(fitted, model_path)
    return model_path


class TestLoadModel:
    """Tests for _load_model function."""

    def test_load_model_returns_pipeline(self, trained_model_path: Path) -> None:
        """Test that _load_model returns a Pipeline object."""
        loaded_model = _load_model(trained_model_path)
        assert isinstance(loaded_model, Pipeline)

    def test_loaded_model_can_predict(
        self, trained_model_path: Path, sample_x: pd.DataFrame
    ) -> None:
        """Test that loaded model can make predictions."""
        loaded_model = _load_model(trained_model_path)
        predictions = loaded_model.predict(sample_x)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_x)
