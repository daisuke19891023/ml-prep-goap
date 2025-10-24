"""Tests for training and prediction actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from goapml.actions.train import Predict, TrainModel
from goapml.models import FileSpec, ModelPolicy, PipelineConfig, WorldState

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Return a minimal pipeline configuration for training actions."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("feature,target\n1,1\n", encoding="utf-8")

    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )

    return PipelineConfig(file=file_spec)


def build_state() -> WorldState:
    """Construct a world state with preprocessed train/test arrays."""
    x_train = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)
    return WorldState(split=(x_train, x_test, y_train, y_test), facts={"features_ready"})


def test_train_model_linear_regression_fits_and_logs(
    pipeline_config: PipelineConfig,
) -> None:
    """Linear regression is fitted and recorded on the state."""
    state = build_state()

    TrainModel().run(state, pipeline_config)

    assert state.model is not None
    from sklearn.linear_model import LinearRegression

    assert isinstance(state.model, LinearRegression)
    assert state.has("trained")
    assert state.logs[-1] == "train_model:linear_regression"


def test_train_model_supports_random_forest(pipeline_config: PipelineConfig) -> None:
    """Random forest models are instantiated using provided parameters."""
    state = build_state()
    model_policy = ModelPolicy(kind="random_forest", params={"n_estimators": 5, "random_state": 0})
    config = pipeline_config.model_copy(update={"model": model_policy})

    TrainModel().run(state, config)

    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(state.model, RandomForestRegressor)
    assert state.has("trained")


def test_predict_generates_predictions_matching_target(
    pipeline_config: PipelineConfig,
) -> None:
    """Predictions align with the validation target size."""
    state = build_state()
    TrainModel().run(state, pipeline_config)

    Predict().run(state, pipeline_config)

    assert state.pred is not None
    assert state.split is not None
    _, _, _, y_test = state.split
    assert len(state.pred) == len(y_test)
    assert state.has("predicted")
    assert state.logs[-1] == "predict"
