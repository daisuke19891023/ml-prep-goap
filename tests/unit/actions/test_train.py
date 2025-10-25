"""Tests for training and prediction actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from numpy.typing import NDArray

from goapml.actions.train import PersistArtifacts, Predict, TrainModel
from goapml.models import (
    ArtifactSpec,
    FileSpec,
    ModelPolicy,
    PipelineConfig,
    WorldState,
)

if TYPE_CHECKING:
    from pathlib import Path

    def _joblib_load(
        filename: str,
        mmap_mode: str | None = None,
        ensure_native_byte_order: str = "auto",
    ) -> Any: ...
else:  # pragma: no cover - runtime import
    from joblib import load as _joblib_load


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

    return PipelineConfig(
        file=file_spec,
        artifacts=ArtifactSpec(directory=str(tmp_path / "artifacts")),
    )


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


def test_persist_artifacts_round_trip_prediction(
    pipeline_config: PipelineConfig,
) -> None:
    """Persisted artefacts can be reloaded to reproduce predictions."""
    x_train_raw = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test_raw = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)

    scaler: Any = StandardScaler()
    x_train_processed = cast(
        NDArray[np.float64],
        np.asarray(scaler.fit_transform(x_train_raw), dtype=float),
    )
    x_test_processed = cast(
        NDArray[np.float64],
        np.asarray(scaler.transform(x_test_raw), dtype=float),
    )

    state = WorldState(
        split=(x_train_processed, x_test_processed, y_train, y_test),
        preprocessor=scaler,
        facts={"features_ready"},
    )

    TrainModel().run(state, pipeline_config)
    Predict().run(state, pipeline_config)

    PersistArtifacts().run(state, pipeline_config)

    assert state.has("persisted")
    assert state.model_path is not None
    assert state.preprocessor_path is not None
    assert state.logs[-1] == "persist_artifacts:model.joblib,preprocessor.joblib"

    loaded_model = _joblib_load(str(state.model_path))
    loaded_preprocessor = _joblib_load(str(state.preprocessor_path))

    transformed_test = cast(
        NDArray[np.float64],
        loaded_preprocessor.transform(x_test_raw),
    )
    reloaded_predictions = cast(
        NDArray[np.float64],
        loaded_model.predict(transformed_test),
    )

    assert state.pred is not None
    baseline_pred = np.asarray(state.pred, dtype=float)
    np.testing.assert_allclose(reloaded_predictions, baseline_pred)


def test_persist_artifacts_requires_preprocessor(
    pipeline_config: PipelineConfig,
) -> None:
    """Persisting artefacts without a preprocessor raises an error."""
    state = build_state()
    TrainModel().run(state, pipeline_config)

    with pytest.raises(RuntimeError):
        PersistArtifacts().run(state, pipeline_config)
