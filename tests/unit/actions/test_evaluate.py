"""Unit tests for the evaluate action."""

from __future__ import annotations

import numpy as np
import pytest

from goapml.actions.evaluate import Evaluate
from goapml.models import FileSpec, PipelineConfig, WorldState


def _build_config(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    path = tmp_path_factory.mktemp("data") / "dummy.csv"
    path.write_text("", encoding="utf-8")
    file_spec = FileSpec(
        path=str(path),
        encoding=None,
        delimiter=",",
        decimal=".",
    )
    return PipelineConfig(file=file_spec)


def test_evaluate_computes_requested_metrics(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Compute configured metrics and ensure sensible scores."""
    config = _build_config(tmp_path_factory)
    state = WorldState()

    x_train = np.zeros((4, 1))
    x_test = np.zeros((4, 1))
    y_train = np.array([0.0, 0.0, 0.0, 0.0])
    y_test = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.05, 1.95, 3.1, 4.05])

    state.split = (x_train, x_test, y_train, y_test)
    state.pred = predictions

    action = Evaluate()
    action.run(state, config)

    assert state.has("evaluated")
    assert state.metrics is not None
    assert set(state.metrics) == {"r2", "rmse", "mae"}
    assert state.metrics["r2"] >= 0.8
    assert state.metrics["rmse"] >= 0.0
    assert state.metrics["mae"] >= 0.0


def test_evaluate_requires_predictions(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Raise an error when predictions are missing."""
    config = _build_config(tmp_path_factory)
    state = WorldState()

    with pytest.raises(RuntimeError):
        Evaluate().run(state, config)
