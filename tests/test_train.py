"""Tests for the animal_shelter.model.train module."""

from pathlib import Path

import pandas as pd
import pytest
from animal_shelter.model.train import (
    _build_pipeline,
    _fit_model,
    _save_model,
)
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


class TestBuildPipeline:
    """Tests for _build_pipeline function."""

    def test_build_pipeline_returns_pipeline(
        self, cat_features: list[str], num_features: list[str]
    ) -> None:
        """Test that _build_pipeline returns a Pipeline object."""
        result = _build_pipeline(cat_features, num_features)
        assert isinstance(result, Pipeline)

    def test_build_pipeline_has_required_steps(
        self, cat_features: list[str], num_features: list[str]
    ) -> None:
        """Test that pipeline has transformer and model steps."""
        pipeline = _build_pipeline(cat_features, num_features)
        assert "transformer" in pipeline.named_steps
        assert "model" in pipeline.named_steps


class TestFitModel:
    """Tests for _fit_model function."""

    def test_fit_model_succeeds(
        self,
        cat_features: list[str],
        num_features: list[str],
        sample_x: pd.DataFrame,
        sample_y: pd.Series,
    ) -> None:
        """Test that _fit_model fits without errors."""
        pipeline = _build_pipeline(cat_features, num_features)
        result = _fit_model(pipeline, sample_x, sample_y)
        assert isinstance(result, Pipeline)


class TestSaveModel:
    """Tests for _save_model function."""

    def test_save_model_creates_file(
        self, cat_features: list[str], num_features: list[str], tmp_path: Path
    ) -> None:
        """Test that _save_model creates a model file."""
        pipeline = _build_pipeline(cat_features, num_features)
        model_path = tmp_path / "model.pkl"
        _save_model(pipeline, model_path)
        assert model_path.exists()

    def test_save_model_creates_parent_directories(
        self, cat_features: list[str], num_features: list[str], tmp_path: Path
    ) -> None:
        """Test that _save_model creates parent directories."""
        pipeline = _build_pipeline(cat_features, num_features)
        model_path = tmp_path / "nested" / "dir" / "model.pkl"
        _save_model(pipeline, model_path)
        assert model_path.exists()
