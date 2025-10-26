"""Train and inference actions for regression models."""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast

import goapml.paths as path_utils

from goapml.schemas import (
    Action,
    ActionSchema,
    PERSIST_ARTIFACTS_SCHEMA,
    PREDICT_SCHEMA,
    TRAIN_MODEL_SCHEMA,
)

if TYPE_CHECKING:
    def _joblib_dump(
        value: Any,
        filename: str | IO[bytes],
        compress: int = 0,
        protocol: Any | None = None,
    ) -> Any: ...

    from sklearn.base import BaseEstimator
    from goapml.models import (
        MODEL_FACTORIES,
        ModelPolicy,
        PipelineConfig,
        PredictionVector,
        WorldState,
    )
else:  # pragma: no cover - runtime fallbacks
    from joblib import dump as _joblib_dump

    from goapml.models import MODEL_FACTORIES

    BaseEstimator = ModelPolicy = PipelineConfig = WorldState = Any
    PredictionVector = Any

__all__ = [
    "PERSIST_ARTIFACTS_SCHEMA",
    "PREDICT_SCHEMA",
    "TRAIN_MODEL_SCHEMA",
    "PersistArtifacts",
    "Predict",
    "TrainModel",
]


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
        try:
            factory = MODEL_FACTORIES[policy.kind]
        except KeyError as exc:  # pragma: no cover - defensive guard
            message = f"Unsupported model kind: {policy.kind}"
            raise ValueError(message) from exc
        return factory(policy.params)


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


@dataclass(slots=True)
class PersistArtifacts(Action):
    """Persist the trained model and fitted preprocessor to disk."""

    schema: ActionSchema = field(default_factory=lambda: PERSIST_ARTIFACTS_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Serialise the trained artefacts using joblib."""
        if state.model is None:
            message = "A trained model is required before persisting artefacts."
            raise RuntimeError(message)
        if state.preprocessor is None:
            message = "A fitted preprocessor is required before persisting artefacts."
            raise RuntimeError(message)

        directory, model_path, preprocessor_path = config.resolve_artifact_paths()

        base_root = Path(config.artifacts_root).resolve()
        try:
            directory = path_utils.ensure_safe_path(base_root, directory)
            model_path = path_utils.ensure_safe_path(base_root, model_path)
            preprocessor_path = path_utils.ensure_safe_path(base_root, preprocessor_path)
        except path_utils.UnsafePathError as exc:
            message = "Failed to persist artefact directory"
            raise RuntimeError(message) from exc

        directory.mkdir(parents=True, exist_ok=True)

        _LOGGER.info(
            "Persisting trained artefacts.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "persist_artifacts",
                "directory": str(directory),
                "model_filename": model_path.name,
                "preprocessor_filename": preprocessor_path.name,
            },
        )

        if not path_utils.secure_open_supported():
            message = (
                "Artifact persistence is unsupported on this platform because "
                "os.O_NOFOLLOW or dir_fd is unavailable."
            )
            raise RuntimeError(message)

        directory_path = os.fspath(directory)
        directory_message = f"Failed to persist artefact directory: {directory}"
        try:
            dir_fd = os.open(
                directory_path,
                os.O_DIRECTORY | os.O_NOFOLLOW | os.O_PATH,
            )
        except OSError as exc:  # pragma: no cover - escalated for observability
            raise RuntimeError(directory_message) from exc

        try:
            try:
                stat_result = os.fstat(dir_fd)
            except OSError as exc:  # pragma: no cover - escalated for observability
                raise RuntimeError(directory_message) from exc

            if not stat.S_ISDIR(stat_result.st_mode):
                raise RuntimeError(directory_message)

            try:
                model_relative = model_path.relative_to(directory)
                preprocessor_relative = preprocessor_path.relative_to(directory)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(directory_message) from exc

            self._dump(
                state.model,
                model_relative,
                dir_fd=dir_fd,
                display_path=model_path,
            )
            self._dump(
                state.preprocessor,
                preprocessor_relative,
                dir_fd=dir_fd,
                display_path=preprocessor_path,
            )
        finally:
            os.close(dir_fd)

        state.model_path = model_path
        state.preprocessor_path = preprocessor_path
        state.add("persisted")
        state.logs.append(
            f"persist_artifacts:{model_path.name},{preprocessor_path.name}",
        )

        _LOGGER.info(
            "Artefact persistence complete.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "persist_artifacts",
            },
        )

    @staticmethod
    def _dump(
        obj: Any,
        path: Path,
        *,
        dir_fd: int,
        display_path: Path,
    ) -> None:
        if not path_utils.secure_open_supported():
            message = (
                "Artifact persistence is unsupported on this platform because "
                "os.O_NOFOLLOW or dir_fd is unavailable."
            )
            raise RuntimeError(message)

        required_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        file_path = os.fspath(path)
        message = f"Failed to persist artefact: {display_path}"

        try:
            fd = os.open(file_path, required_flags, 0o600, dir_fd=dir_fd)
        except OSError as exc:  # pragma: no cover - escalated for observability
            raise RuntimeError(message) from exc

        try:
            file_obj = cast("IO[bytes]", os.fdopen(fd, "wb"))
        except OSError as exc:  # pragma: no cover - escalated for observability
            os.close(fd)
            raise RuntimeError(message) from exc

        with file_obj:
            try:
                _joblib_dump(obj, file_obj)
            except OSError as exc:  # pragma: no cover - escalated for observability
                raise RuntimeError(message) from exc
