"""Evaluation actions computing regression metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from goapml.models import PipelineConfig, WorldState
else:  # pragma: no cover - runtime fallback types
    PipelineConfig = WorldState = Any

__all__ = ["EVALUATE_SCHEMA", "Evaluate"]

MetricFunc = Callable[[Any, Any], float]
FloatArray = NDArray[np.float64]


def _ensure_array(values: Any) -> FloatArray:
    array = np.asarray(values, dtype=float)
    return np.ravel(array)


def _rmse(y_true: Any, y_pred: Any) -> float:
    true = _ensure_array(y_true)
    pred = _ensure_array(y_pred)
    diff = true - pred
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(y_true: Any, y_pred: Any) -> float:
    true = _ensure_array(y_true)
    pred = _ensure_array(y_pred)
    return float(np.mean(np.abs(true - pred)))


def _r2(y_true: Any, y_pred: Any) -> float:
    true = _ensure_array(y_true)
    pred = _ensure_array(y_pred)
    diff = true - pred
    ss_res = float(np.sum(diff * diff))
    true_mean = float(np.mean(true))
    centered = true - true_mean
    ss_tot = float(np.sum(centered * centered))
    if ss_tot == 0.0:
        return 1.0 if np.allclose(true, pred) else 0.0
    return float(1.0 - (ss_res / ss_tot))


EVALUATE_SCHEMA = ActionSchema(
    name="evaluate",
    requires={"predicted"},
    provides={"evaluated"},
    cost=1.0,
)


@dataclass(slots=True)
class Evaluate(Action):
    """Compute evaluation metrics for regression predictions."""

    schema: ActionSchema = field(default_factory=lambda: EVALUATE_SCHEMA)

    _METRICS: ClassVar[Mapping[str, MetricFunc]] = {
        "r2": _r2,
        "rmse": _rmse,
        "mae": _mae,
    }

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Compute configured metrics and store them on the world state."""
        if state.pred is None:
            message = "Predictions must be available before evaluation."
            raise RuntimeError(message)
        if state.split is None:
            message = "Train/test split must be available for evaluation."
            raise RuntimeError(message)

        _, _, _, y_test = state.split
        metrics_to_compute = config.eval.metrics

        metrics: dict[str, float] = {}
        for metric_name in metrics_to_compute:
            try:
                metric_func = self._METRICS[metric_name]
            except KeyError as exc:  # pragma: no cover - defensive guard
                message = f"Unsupported metric requested: {metric_name}"
                raise ValueError(message) from exc
            value = float(metric_func(y_test, state.pred))
            metrics[metric_name] = value

        state.metrics = metrics
        state.add("evaluated")
        state.logs.append(f"evaluate:{','.join(metrics_to_compute)}")
