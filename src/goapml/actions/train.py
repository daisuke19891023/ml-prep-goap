"""Train and inference actions for regression models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from goapml.schemas import (
    Action,
    ActionSchema,
    PREDICT_SCHEMA,
    TRAIN_MODEL_SCHEMA,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from goapml.models import ModelPolicy, PipelineConfig, PredictionVector, WorldState
else:  # pragma: no cover - runtime fallbacks
    BaseEstimator = ModelPolicy = PipelineConfig = WorldState = Any
    PredictionVector = Any

__all__ = ["PREDICT_SCHEMA", "TRAIN_MODEL_SCHEMA", "Predict", "TrainModel"]


_LOGGER = logging.getLogger(__name__)


class _RegressorProtocol(Protocol):
    """Structural protocol for estimators supporting fit and predict."""

    def fit(self, x: Any, y: Any) -> Any: ...

    def predict(self, x: Any) -> PredictionVector: ...


@dataclass(slots=True)
class TrainModel(Action):
    """Fit the configured regression model on the prepared training data."""

    schema: ActionSchema = field(default_factory=lambda: TRAIN_MODEL_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Instantiate and fit the model, recording it on the world state."""
        if state.split is None:
            message = "Features must be prepared before training the model."
            raise RuntimeError(message)

        _LOGGER.info(
            "Training regression model.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "train_model",
                "model": config.model.kind,
            },
        )
        x_train, x_test, y_train, y_test = state.split
        model = self._create_model(config.model)
        regressor = cast("_RegressorProtocol", model)
        regressor.fit(x_train, y_train)

        state.model = model
        state.split = (x_train, x_test, y_train, y_test)
        state.add("trained")
        state.logs.append(f"train_model:{config.model.kind}")
        _LOGGER.info(
            "Model training complete.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "train_model",
                "model": config.model.kind,
            },
        )

    def _create_model(self, policy: ModelPolicy) -> BaseEstimator:
        """Return the estimator configured by ``policy``."""
        if policy.kind == "linear_regression":
            return LinearRegression(**policy.params)
        if policy.kind == "random_forest":
            return RandomForestRegressor(**policy.params)
        message = f"Unsupported model kind: {policy.kind}"
        raise ValueError(message)


@dataclass(slots=True)
class Predict(Action):
    """Generate predictions from the trained model for the validation set."""

    schema: ActionSchema = field(default_factory=lambda: PREDICT_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # noqa: ARG002
        """Compute predictions ensuring they align with the validation target."""
        if state.model is None:
            message = "A trained model is required before running predictions."
            raise RuntimeError(message)
        if state.split is None:
            message = "Train/test split must be available to make predictions."
            raise RuntimeError(message)

        _LOGGER.info(
            "Generating predictions.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "predict",
            },
        )
        _, x_test, _, y_test = state.split
        model = cast("_RegressorProtocol", state.model)
        predictions = model.predict(x_test)

        if len(predictions) != len(y_test):
            message = "Prediction count must match validation target length."
            raise RuntimeError(message)

        state.pred = predictions
        state.add("predicted")
        state.logs.append("predict")
        _LOGGER.info(
            "Predictions generated.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "predict",
                "observations": len(y_test),
            },
        )
